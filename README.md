![Dream Textures, subtitle: Stable Diffusion built-in to Blender](docs/assets/banner.png)

[![Latest Release](https://flat.badgen.net/github/release/carson-katri/dream-textures)](https://github.com/carson-katri/dream-textures/releases/latest)
[![Join the Discord](https://flat.badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/EmDJ8CaWZ7)
[![Total Downloads](https://img.shields.io/github/downloads/carson-katri/dream-textures/total?style=flat-square)](https://github.com/carson-katri/dream-textures/releases/latest)
[![Buy on Blender Market](https://flat.badgen.net/badge/buy/blender%20market/orange)](https://www.blendermarket.com/products/dream-textures)

* Create textures, concept art, background assets, and more with a simple text prompt
* Use the 'Seamless' option to create textures that tile perfectly with no visible seam
* Texture entire scenes with 'Project Dream Texture' and depth to image
* Re-style animations with the Cycles render pass
* Run the models on your machine to iterate without slowdowns from a service

# Installation
Download the [latest release](https://github.com/carson-katri/dream-textures/releases/latest) and follow the instructions there to get up and running.

> On macOS, it is possible you will run into a quarantine issue with the dependencies. To work around this, run the following command in the app `Terminal`: `xattr -r -d com.apple.quarantine ~/Library/Application\ Support/Blender/3.3/scripts/addons/dream_textures/.python_dependencies`. This will allow the PyTorch `.dylib`s and `.so`s to load without having to manually allow each one in System Preferences.

If you want a visual guide to installation, see this video tutorial from Ashlee Martino-Tarr: https://youtu.be/kEcr8cNmqZk
> Ensure you always install the [latest version](https://github.com/carson-katri/dream-textures/releases/latest) of the add-on if any guides become out of date.

# Usage

Here's a few quick guides:

## [Setting Up](https://github.com/carson-katri/dream-textures/wiki/Setup)
Setup instructions for various platforms and configurations.

## [Image Generation](https://github.com/carson-katri/dream-textures/wiki/Image-Generation)
Create textures, concept art, and more with text prompts. Learn how to use the various configuration options to get exactly what you're looking for.

![A graphic showing each step of the image generation process](docs/assets/image_generation.png)

## [Texture Projection](https://github.com/carson-katri/dream-textures/wiki/Texture-Projection)
Texture entire models and scenes with depth to image.

![A graphic showing each step of the texture projection process](docs/assets/texture_projection.png)

## [Inpaint/Outpaint](https://github.com/carson-katri/dream-textures/wiki/Inpaint-and-Outpaint)
Inpaint to fix up images and convert existing textures into seamless ones automatically.

Outpaint to increase the size of an image by extending it in any direction.

![A graphic showing each step of the outpainting process](docs/assets/inpaint_outpaint.png)

## [Render Engine](https://github.com/carson-katri/dream-textures/wiki/Render-Engine)
Use the Dream Textures node system to create complex effects.

![A graphic showing each frame of a render, split with the scene and generated result](docs/assets/render_pass.png)

## [AI Upscaling](https://github.com/carson-katri/dream-textures/wiki/AI-Upscaling)
Upscale your low-res generations 4x.

![A graphic showing each step of the upscaling process](docs/assets/upscale.png)

## [History](https://github.com/carson-katri/dream-textures/wiki/History)
Recall, export, and import history entries for later use.

# Compatibility
Dream Textures has been tested with CUDA and Apple Silicon GPUs. Over 4GB of VRAM is recommended.

If you have an issue with a supported GPU, please create an issue.

### Cloud Processing
If your hardware is unsupported, you can use DreamStudio to process in the cloud. Follow the instructions in the release notes to setup with DreamStudio.

# Contributing
For detailed instructions on installing from source, see the guide on [setting up a development environment](https://github.com/carson-katri/dream-textures/wiki/Setting-Up-a-Development-Environment).

# Troubleshooting

If you are experiencing trouble getting Dream Textures running, check Blender's system console (in the top left under the "Window" dropdown next to "File" and "Edit") for any error messages. Then [search in the issues list](https://github.com/carson-katri/dream-textures/issues?q=is%3Aissue) with your error message and symptoms.

> **Note** On macOS there is no option to open the system console. Instead, you can get logs by opening the app *Terminal*, entering the command `/Applications/Blender.app/Contents/MacOS/Blender` and pressing the Enter key. This will launch Blender and any error messages will show up in the Terminal app.

![A screenshot of the "Window" > "Toggle System Console" menu action in Blender](docs/assets/readme-toggle-console.png)

Features and feedback are also accepted on the issues page. If you have any issues that aren't listed, feel free to add them there!

The [Dream Textures Discord server](https://discord.gg/EmDJ8CaWZ7) also has a common issues list and strong community of helpful people, so feel free to come by for some help there as well.



## How to setup the script based image generation

You can generate images using Dream Textures without Blender by running a Python script directly. This is useful for automation, headless servers, or integrating with other tools.

### 1. Install requirements

Make sure you have Python 3.8+ and install the required packages:

```sh
pip install diffusers torch numpy pillow
```

### 2. Prepare the script

Use the provided script at `tests/__main__.py` as a template. Example:

```python
from PIL import Image
from dream_textures.generator_process.actions.prompt_to_image import prompt_to_image
from dream_textures.generator_process.models import Optimizations, Scheduler

prompt = "A fantasy castle on a hill"
model = "CompVis/stable-diffusion-v1-4"  # or another supported model
output_path = "output.png"
steps = 30
width = 512
height = 512
seed = 42
cfg_scale = 7.5

optimizations = Optimizations()
scheduler = Scheduler("DDIM")

gen = prompt_to_image(
    None,
    model=model,
    scheduler=scheduler,
    optimizations=optimizations,
    prompt=prompt,
    steps=steps,
    width=width,
    height=height,
    seed=seed,
    cfg_scale=cfg_scale,
    use_negative_prompt=False,
    negative_prompt="",
    seamless_axes=None,
    iterations=1,
    step_preview_mode=None
)

future = next(gen)
future.set_done()
result = future.result()

img_np = result[0] if isinstance(result, list) and len(result) > 0 else result
img = Image.fromarray(img_np.astype("uint8"))
img.save(output_path)
print(f"Image saved to {output_path}")
```

### 3. Run the script

From the project root directory, run:

```sh
PYTHONPATH=. python tests/__main__.py
```

Or, if you want to use the module runner:

```sh
PYTHONPATH=. python -m tests
```

### 4. Output

The generated image will be saved as `output.png` in your current directory.

**Tip:**
- For SD 2.1-base, use 768x768 resolution. For v1 models, use 512x512.
- You can change the prompt, model, and other parameters in the script.

**Important:**
- Before running the script, set the Blender version environment variable:
  ```sh
  export BLENDER_VERSION=4.5.0
  ```
- Always run the script from the project root directory (not from inside the `tests` folder), so Python can find the `dream_textures` package.
