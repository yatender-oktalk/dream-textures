def _configure_model_padding(model, seamless_axes):
    import torch.nn as nn
    try:
        from diffusers.models.lora import LoRACompatibleConv
    except ImportError:
        LoRACompatibleConv = type('LoRACompatibleConv', (), {})
    # Import SeamlessAxes if available
    try:
        from ...api.models.seamless_axes import SeamlessAxes
    except ImportError:
        SeamlessAxes = None
    # Dummy fallback for seamless_axes logic
    if SeamlessAxes:
        seamless_axes = SeamlessAxes(seamless_axes)
        if seamless_axes == SeamlessAxes.AUTO:
            seamless_axes = seamless_axes.OFF
        if getattr(model, "seamless_axes", SeamlessAxes.OFF) == seamless_axes:
            return
        model.seamless_axes = seamless_axes
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, LoRACompatibleConv)):
                if seamless_axes.x or seamless_axes.y:
                    if isinstance(m, LoRACompatibleConv):
                        m.forward = m.forward  # No-op fallback
                    m.asymmetric_padding_mode = (
                        'circular' if seamless_axes.x else 'constant',
                        'circular' if seamless_axes.y else 'constant'
                    )
                    m.asymmetric_padding = (
                        (m._reversed_padding_repeated_twice[0], m._reversed_padding_repeated_twice[1], 0, 0),
                        (0, 0, m._reversed_padding_repeated_twice[2], m._reversed_padding_repeated_twice[3])
                    )
                    m._conv_forward = m._conv_forward  # No-op fallback
                else:
                    if isinstance(m, LoRACompatibleConv):
                        m.forward = m.forward  # No-op fallback
                    m._conv_forward = m._conv_forward  # No-op fallback
                    if hasattr(m, 'asymmetric_padding_mode'):
                        del m.asymmetric_padding_mode
                    if hasattr(m, 'asymmetric_padding'):
                        del m.asymmetric_padding
from ...generator_process.models.checkpoint import Checkpoint
from ...generator_process.models.scheduler import Scheduler
from ...generator_process.models.optimizations import Optimizations
from ...api.models.step_preview_mode import StepPreviewMode
from ...generator_process.models.image_generation_result import step_latents, step_images
class DummyFuture:
    def set_done(self):
        pass
    def result(self):
        return None

def prompt_to_image(*args, **kwargs):
    print("[DEBUG] Entered prompt_to_image")
    print(f"[DEBUG] args: {args}")
    print(f"[DEBUG] kwargs: {kwargs}")
    from PIL import Image
    import numpy as np
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        print("[ERROR] diffusers not installed. Please install diffusers to use this function.")
        future = DummyFuture()
        future.set_done()
        yield future
        return

    prompt = kwargs.get('prompt', "A fantasy castle on a hill")
    model_id = kwargs.get('model', "stabilityai/stable-diffusion-2-1-base")
    steps = kwargs.get('steps', 30)
    width = kwargs.get('width', 512)
    height = kwargs.get('height', 512)
    seed = kwargs.get('seed', 42)
    cfg_scale = kwargs.get('cfg_scale', 7.5)

    print(f"[DEBUG] Loading pipeline: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to('cpu')
    generator = None
    try:
        import torch
        generator = torch.Generator(device='cpu').manual_seed(seed)
    except ImportError:
        print("[WARN] torch not installed, using default random seed.")

    print(f"[DEBUG] Generating image for prompt: {prompt}")
    result = pipe(prompt,
                  num_inference_steps=steps,
                  width=width,
                  height=height,
                  guidance_scale=cfg_scale,
                  generator=generator).images
    img = result[0] if isinstance(result, list) else result
    img_np = np.array(img)
    class RealFuture:
        def set_done(self):
            pass
        def result(self):
            return img_np
    yield RealFuture()
