import sys
import cgitb
import json
from pathlib import Path
import argparse, os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
import pandas as pd
logging.set_verbosity_error()

# Opening JSON file
directory = ".SD_Temp"
parentdir = str(Path.home())
filename = "sample.json"
temppath = os.path.join(parentdir,directory)
Path(temppath).mkdir(parents=True, exist_ok=True)
filepath = os.path.join(temppath, filename)
with open(filepath,'r') as openfile:
        json_object = json.load(openfile)

#get elements from json
input_image = json_object.get('input_image')
prompt = json_object.get('prompt')

ddim_steps = json_object.get('ddim_steps')
n_iter = json_object.get("n_iter")
strength = json_object.get("strength")
H = json_object.get('h')
W = json_object.get('w')
n_samples = json_object.get('n_samples')
#n_rows = json_object.get('n_rows')
seed = json_object.get('seed')

outdir = json_object.get('outdir')
format = json_object.get('format')
precision = json_object.get('precision')
sampler = json_object.get('sampler')


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(path, h0, w0):

    image = Image.open(input_image).convert("RGB")
    w, h = image.size

    print(f"loaded input image of size ({w}, {h}) from {path}")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


config = "C:/StableDiffusion/InvokeAI/optimizedSD/v1-inference.yaml"
DEFAULT_CKPT = "C:/StableDiffusion/InvokeAI/models/ldm/stable-diffusion-v1/model.ckpt"

#add default values
ckpt = DEFAULT_CKPT
unet_bs = 1
device = "cuda"
turbo = True
fixed_code = True
skip_grid = True
skip_save = True
sn_rows = 0
from_file = ""
ddim_eta = 0.0
n_iter = 1
n_rows = 0
scale = 7.5

tic = time.time()
os.makedirs(outdir, exist_ok=True)
outpath = outdir
grid_count = len(os.listdir(outpath)) - 1

if seed == None:
    seed = randint(0, 1000000)
seed_everything(seed)

# Logging
logger(json_object, log_csv = "logs/img2img_logs.csv")

sd = load_model_from_config(f"{ckpt}")
li, lo = [], []
for key, value in sd.items():
    sp = key.split(".")
    if (sp[0]) == "model":
        if "input_blocks" in sp:
            li.append(key)
        elif "middle_block" in sp:
            li.append(key)
        elif "time_embed" in sp:
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd["model1." + key[6:]] = sd.pop(key)
for key in lo:
    sd["model2." + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")

assert os.path.isfile(input_image)
init_image = load_img(input_image, H, W).to(device)

model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
model.cdevice = device
model.unet_bs = unet_bs
model.turbo = turbo

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = device

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd
if device != "cpu" and precision == "autocast":
    model.half()
    modelCS.half()
    modelFS.half()
    init_image = init_image.half()

batch_size = n_samples
n_rows = n_rows if n_rows > 0 else batch_size
if not from_file:
    assert prompt is not None
    prompt = prompt
    data = [batch_size * [prompt]]

else:
    print(f"reading prompts from {from_file}")
    with open(from_file, "r") as f:
        data = f.read().splitlines()
        data = batch_size * list(data)
        data = list(chunk(sorted(data), batch_size))

modelFS.to(device)

init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space

if device != "cpu":
    mem = torch.cuda.memory_allocated(device=device) / 1e6
    modelFS.to("cpu")
    while torch.cuda.memory_allocated(device=device) / 1e6 >= mem:
        time.sleep(1)


assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
t_enc = int(strength * ddim_steps)
print(f"target t_enc is {t_enc} steps")


if precision == "autocast" and device != "cpu":
    precision_scope = autocast
else:
    precision_scope = nullcontext

print(data)

seeds = ""
with torch.no_grad():

    all_samples = list()
    for n in trange(n_iter, desc="Sampling"):
        for prompts in tqdm(data, desc="data"):

            sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompts[0])))[:150]
            os.makedirs(sample_path, exist_ok=True)
            base_count = len(os.listdir(sample_path))

            with precision_scope("cuda"):
                modelCS.to(device)
                uc = None
                if scale != 1.0:
                    uc = modelCS.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)

                subprompts, weights = split_weighted_subprompts(prompts[0])
                if len(subprompts) > 1:
                    c = torch.zeros_like(uc)
                    totalWeight = sum(weights)
                    # normalize each "sub prompt" and add it
                    for i in range(len(subprompts)):
                        weight = weights[i]
                        # if not skip_normalize:
                        weight = weight / totalWeight
                        c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                else:
                    c = modelCS.get_learned_conditioning(prompts)

                if device != "cpu":
                    mem = torch.cuda.memory_allocated(device=device) / 1e6
                    modelCS.to("cpu")
                    while torch.cuda.memory_allocated(device=device) / 1e6 >= mem:
                        time.sleep(1)

                # encode (scaled latent)
                z_enc = model.stochastic_encode(
                    init_latent,
                    torch.tensor([t_enc] * batch_size).to(device),
                    seed,
                    ddim_eta,
                    ddim_steps,
                )
                # decode it
                samp_ddim = model.sample(
                    t_enc,
                    c,
                    z_enc,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    sampler = sampler
                )

                modelFS.to(device)
                print("saving images")
                for i in range(batch_size):

                    x_samples_ddim = modelFS.decode_first_stage(samp_ddim[i].unsqueeze(0))
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.{format}")
                    )
                    seeds += str(seed) + ","
                    seed += 1
                    base_count += 1

                if device != "cpu":
                    mem = torch.cuda.memory_allocated(device=device) / 1e6
                    modelFS.to("cpu")
                    while torch.cuda.memory_allocated(device=device) / 1e6 >= mem:
                        time.sleep(1)

                del samp_ddim
                print("memory_final = ", torch.cuda.memory_allocated(device=device) / 1e6)

toc = time.time()

time_taken = (toc - tic) / 60.0

print(
    (
        "Samples finished in {0:.2f} minutes and exported to "
        + sample_path
        + "\n Seeds used = "
        + seeds[:-1]
    ).format(time_taken)
)
