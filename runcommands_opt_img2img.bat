@echo off

:: path to your miniconda

call C:\Users\Pin\anaconda3\Scripts\activate.bat

call conda activate ldm

:: path to your stable diffusion

cd C:/StableDiffusion/InvokeAI/

::path to your ldm environment and path to the script

C:\Users\Pin\anaconda3\envs\ldm\python.exe "C:\StableDiffusion\InvokeAI\optimizedSD\optimized_img2img_SP.py"

pause


