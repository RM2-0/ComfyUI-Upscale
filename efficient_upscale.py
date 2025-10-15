import torch
import comfy.utils
import comfy.model_management as model_management
from comfy_extras.nodes_upscale_model import UpscaleModelLoader

class BatchImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "upscale_model": ("UPSCALE_MODEL",),
            "image": ("IMAGE",),
            "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048}),
            "use_fp16": ("BOOLEAN", {"default": True}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image, tile_size=512, use_fp16=True):
        device = model_management.get_torch_device()
        dtype = torch.float16 if use_fp16 else torch.float32

        # Calculate memory requirements
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (tile_size * tile_size * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        # Move model to device and set precision
        upscale_model.to(device)
        if use_fp16:
            upscale_model.half()
        else:
            upscale_model.float()

        # Prepare input images
        in_img = image.movedim(-1, -3).to(device, dtype=dtype)

        # Process with tiling
        overlap = min(32, tile_size // 8)  # Adaptive overlap

        def process_function(x):
            with torch.cuda.amp.autocast(enabled=use_fp16):
                return upscale_model(x)

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], 
                                                                          tile_x=tile_size, tile_y=tile_size, 
                                                                          overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                
                # Use the built-in tiled_scale function which handles batching efficiently
                s = comfy.utils.tiled_scale(
                    in_img,
                    process_function,
                    tile_x=tile_size,
                    tile_y=tile_size,
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    out_channels=in_img.shape[1],
                    output_device=device,
                    pbar=pbar
                )
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile_size //= 2
                overlap = min(32, tile_size // 8)
                if tile_size < 128:
                    raise e

        # Move model back to CPU and clean up
        upscale_model.to("cpu")
        if use_fp16:
            upscale_model.float()

        # Format output
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)

NODE_CLASS_MAPPINGS = {
    "BatchImageUpscaleWithModel": BatchImageUpscaleWithModel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageUpscaleWithModel": "Efficient Upscale (FP16)"
}