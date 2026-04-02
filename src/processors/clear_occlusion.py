##################################################################
# Author: anh-nn01; March 08, 2026
# Description: 
#   Generate occlusion-free 2D images
##################################################################

import sys
import os
import argparse
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{pwd}/../../ObjectClear/")

import glob
import torch
from objectclear.pipelines import ObjectClearPipeline
from objectclear.utils import resize_by_short_side
from PIL import Image
import numpy as np

from generate_semantics import Semantic2DGenerator
sys.path.append(f"{pwd}/../configs")
import cfg_bev
from cfg_bev import OCCLUSION_CLASSES, SEG_THRESHOLD

import time



class OcclusionFreeGenerator:
    def __init__(self, 
        use_fp16=True, seed=42, 
        steps=20, guidance_scale=2.5, no_agf=False,
        cache_dir=None
    ):
        """Initialize parameters: 
            based on https://github.com/zjx0101/ObjectClear/blob/main/inference_objectclear.py
        """
        self.use_fp16 = use_fp16
        self.seed = seed
        self.steps = steps # diffusion steps
        self.guidance_scale = guidance_scale # CFG
        self.no_agf = no_agf # Disable Attention Guided Fusion
        self.cache_dir = cache_dir

        # ObjectClear pipeline
        self.torch_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.variant = "fp16" if self.use_fp16 else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.use_agf = not self.no_agf
        self.pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
            "jixin0101/ObjectClear",
            torch_dtype=self.torch_dtype,
            apply_attention_guided_fusion=self.use_agf,
            cache_dir=self.cache_dir,
            variant=self.variant,
        )
        self.pipe.to(self.device)

    def clear_2d_occlusion(self, images, masks):
        """ 
        Args:
            images: list of images 
            masks:  list of occlusion masks; associated with each image
            export_path

        Outputs:
            occ_free_images: list of occlusion-free images
        """
        occ_free_images = []
        for i, (image, mask) in enumerate(zip(images, masks)):
            # ObjectClear model was trained on 512×512 resolution.
            # Resizing the input so that the **shorter side is 512** helps achieve the best performance.
            image = resize_by_short_side(image, 512, resample=Image.BICUBIC)
            mask = resize_by_short_side(mask, 512, resample=Image.NEAREST)
            w, h = image.size

            result = self.pipe(
                prompt="remove the instance of object",
                image=image,
                mask_image=mask,
                generator=self.generator,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                height=h,
                width=w,
                return_attn_map=False,
            )

            fused_img_pil = result.images[0]
            occ_free_images.append(fused_img_pil)
        
        
        return occ_free_images


""" This is for testing on a single image only
    To process on the whole folder, import the above functions

    Example:
    python clear_occlusion.py \
    --img_path=../examples/nusc/img_4/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151605512404.jpg \
    --export_path=../examples/nusc/img_4/
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate occlusion-free image")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--export_path", type=str, default=None, help="Directory to save output occlusion-free image")
    args = parser.parse_args()

    # input image
    image = Image.open(args.img_path).convert("RGB")
    
    # 1. initialize occlusion free image generator
    generator_clear = OcclusionFreeGenerator(steps=cfg_bev.DIFFUSION_STEPS)
    # 2. initialize occlusion mask generator
    generator_sam3 = Semantic2DGenerator(threshold=SEG_THRESHOLD)
    # 3. occlusion masking
    start = time.time()
    occ_masks = generator_sam3.generate_2Dsem(
		image, semantic_classes=OCCLUSION_CLASSES
	)
    occ_mask = np.logical_or.reduce(list(occ_masks.values()))
    occ_mask = (occ_mask * 255).astype(np.uint8) # Convert to 0-255 uint8 format
    occ_mask = occ_mask[0] # (1,H,W) => (H,W)
    occ_mask = Image.fromarray(occ_mask, mode='L')
    end = time.time()
    print('Total SAM3 runtime / image:', round(end-start,3), 'seconds.')
    
    # 4. generate occlusion-free image and save to export_path
    images = [image]
    masks = [occ_mask]
    start = time.time()
    occ_free_images = generator_clear.clear_2d_occlusion(images, masks)
    end = time.time()
    print('Total ObjectClear runtime / image:', round(end-start,3), 'seconds.')
    img_out = occ_free_images[0]
    # save image
    os.makedirs(args.export_path, exist_ok=True)
    img_out.save(os.path.join(args.export_path, "img_occlusion_free.jpg"))