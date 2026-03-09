##################################################################
# Author: anh-nn01; Feb 09, 2026
# Description: 
#   Generate 2D semantics from monocular image
##################################################################

import sys
import os
import argparse
pwd = os.path.dirname(os.path.abspath(__file__))

import torch
import numpy as np
from PIL import Image

sys.path.append(f"{pwd}/../../sam3/")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sys.path.append(f"{pwd}/../configs")
from cfg_bev import SEMANTIC_CLASSES


class Semantic2DGenerator:
	def __init__(self, threshold=0.15):
		#############################
		# Model initialization
		#############################
		self.model = build_sam3_image_model()
		self.processor = Sam3Processor(self.model)
		self.threshold = threshold

	def generate_2Dsem(self, image, semantic_classes=SEMANTIC_CLASSES):
		inference_state = self.processor.set_image(image)
		masks = {}
		for prompt in semantic_classes:
			output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
			# Get the masks, bounding boxes, and scores
			# masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
			# sum over all individual object masks
			masks[prompt] = output["masks"].sum(0).cpu().numpy()
			masks[prompt] = (masks[prompt] > self.threshold).astype(np.uint8) * 255

		return masks

""" This is for testing on a single image only
	To process on the whole folder, import the above functions

	python generate_semantics.py \
	--img_path= \
	--export_path= \
"""
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate 2D semantics from monocular image")
	parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
	parser.add_argument("--export_path", type=str, required=True, help="Directory to save output semantic masks")
	args = parser.parse_args()

	# generate 2D semantics and save to export_path
	generator = Semantic2DGenerator()
	image = Image.open(args.img_path)
	masks = generator.generate_2Dsem(
		image, semantic_classes=SEMANTIC_CLASSES
	)
	# save masks to export_dir
	os.makedirs(args.export_path, exist_ok=True)
	np.savez(os.path.join(args.export_path, "semantic_masks.npz"), **masks)
