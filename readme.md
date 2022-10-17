# SD_img2img_hda

This is inspired by [WaffleBoyTom](https://github.com/WaffleBoyTom/StableDiffusionIntegration.git).

So that the method is the same, just making the other StableDiffusion img2img to excute and view in Houdini.

Since I've been used to use Redshift, the hda is made for it. You can reference any image path into the Input Image parameter.

Please read the instruction of [WaffleBoyTom](https://github.com/WaffleBoyTom/StableDiffusionIntegration.git) first.

Here are steps I suggest to use my img2img hda, very much like [WaffleBoyTom](https://github.com/WaffleBoyTom/StableDiffusionIntegration.git).
1. Download optimizedSD directory from [basujindal/stable-diffusion](https://github.com/basujindal/stable-diffusion.git) to your StableDiffusion root.
2. Put optimized_img2img_SP.py into optimizedSD directory.
3. Put bat files to anywhere then change file path in the bat files.
4. Change the file path in the python module of the hda in the runSD() function.
5. Bring in SD_img2img hda at /out in Houdini.
6. "fetch images" button would create a cop2net in /out and images are in it.
7. https://youtu.be/JjICpRA_dlg

The script reads png/jpg image only, remenber to render png/jpg format image.

If you wonder why there is only optimized img2img python script.
My graphic can't handle original scripts, so that I can only test with optimized version.


