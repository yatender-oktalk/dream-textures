
# Standalone script to generate an image using Dream Textures backend (no Blender required)
import sys
import numpy as np
from PIL import Image
from dream_textures.generator_process.actions.prompt_to_image import prompt_to_image
from dream_textures.generator_process.models import Optimizations, Scheduler

# Example arguments (adjust as needed)
prompt = "A fantasy castle on a hill"
model = "CompVis/stable-diffusion-v1-4"
output_path = "output.png"
steps = 30
width = 512
height = 512
seed = 42
cfg_scale = 7.5
use_negative_prompt = False
negative_prompt = ""
seamless_axes = None
iterations = 1
step_preview_mode = None

# Set up dummy optimizations and scheduler (customize as needed)
optimizations = Optimizations()
scheduler = Scheduler("DDIM")

# Call the prompt_to_image generator function
gen = prompt_to_image(
	None,  # self (not used)
	model=model,
	scheduler=scheduler,
	optimizations=optimizations,
	prompt=prompt,
	steps=steps,
	width=width,
	height=height,
	seed=seed,
	cfg_scale=cfg_scale,
	use_negative_prompt=use_negative_prompt,
	negative_prompt=negative_prompt,
	seamless_axes=seamless_axes,
	iterations=iterations,
	step_preview_mode=step_preview_mode
)

# prompt_to_image yields a Future; get the result
future = next(gen)
future.set_done()  # Ensure the future is marked done if not already
result = future.result()

print("Generation result:", result)

# result is a list of images (numpy arrays); take the first
img_np = None
if isinstance(result, list) and len(result) > 0:
	img_np = result[0]
elif result is not None:
	img_np = result

if img_np is None:
	print("No image was generated. Check model weights, dependencies, and backend logs.")
	exit(1)

# Convert numpy array to PIL Image and save
img = Image.fromarray(np.uint8(img_np))
img.save(output_path)
print(f"Image saved to {output_path}")