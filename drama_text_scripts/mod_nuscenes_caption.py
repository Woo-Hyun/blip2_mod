import torch
from PIL import Image, ImageDraw
from lavis.models import load_model_and_preprocess
import os
import json
import argparse
from tqdm import tqdm
import time

from nuscenes.nuscenes import NuScenes
from data_converter.trajectory_api import NuScenesTraj

import logging
from packaging import version

from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
import transformers

from lavis.models.blip2_models.blip2_opt import Blip2OPT
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer

def concatenate_images(original, cropped):
    crop_width, crop_height = cropped.size
    orig_width, orig_height = original.size
    
    if crop_height < crop_width:
        if orig_width == 0 or int(crop_height * (orig_width / crop_width)) == 0:
            return original
        cropped_resized = cropped.resize((orig_width, int(crop_height * (orig_width / crop_width))))
        
        concatenated = Image.new('RGB', (orig_width, orig_height + cropped_resized.height))
        concatenated.paste(original, (0, 0))
        concatenated.paste(cropped_resized, (0, orig_height))
    else:
        if int(crop_width * (orig_height / crop_height)) == 0 or orig_height == 0:
            return original
        cropped_resized = cropped.resize((int(crop_width * (orig_height / crop_height)), orig_height))
        
        concatenated = Image.new('RGB', (orig_width + cropped_resized.width, orig_height))
        concatenated.paste(original, (0, 0))
        concatenated.paste(cropped_resized, (orig_width, 0))
    
    return concatenated

NameMapping = {
        'vehicle.bicycle',
        'vehicle.bus.bendy',
        'vehicle.bus.rigid',
        'vehicle.car',
        'vehicle.construction',
        'vehicle.motorcycle',
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'human.pedestrian.construction_worker',
        'human.pedestrian.police_officer',
        'vehicle.trailer',
        'vehicle.truck'
}

def main(nuscenes_dir: str, json_dir: str):

    ##### loads BLIP-2 pre-trained model #####
    device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)

    fine_tuned_weights_path = "/workspace/blip2_mod/lavis/output/BLIP2/Caption_coco/20231023073_opt_seed2023/checkpoint_best.pth"
    checkpoint = torch.load(fine_tuned_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)

    with open(json_dir, "r") as f:
        data = json.load(f)

    new_data = {}
    for image_path, object_list in tqdm(data.items(), desc="Processing images", position=0):
        image = Image.open(os.path.join(nuscenes_dir, image_path)).convert("RGB")
        
        new_obj_list = []
        for obj in tqdm(object_list, desc=f"Processing objects in {image_path}", position=1, leave=False):
            visibility = int(obj["visibility_token"])
            x_min, y_min, x_max, y_max = obj["bbox_corners"]
            width = x_max - x_min
            height = y_max - y_min
            category = obj["category_name"]

            if (visibility == 3 or visibility == 4) and (width >= 100 or height >= 100) and (category in NameMapping) and (width >= 20 and height >= 20):
                cropped_image = image.crop(tuple(obj["bbox_corners"]))
                merged_image = concatenate_images(image, cropped_image)

                merged_image = vis_processors["eval"](merged_image).unsqueeze(0).to(device)
                text = model.generate({"image": merged_image})
                obj["caption_seed2023"] = text

            new_obj_list.append(obj)
        
        new_data[image_path] = new_obj_list

    new_json_dir = "/workspace/blip2_mod/docs/_static/nuscenes_test.json"
    with open(new_json_dir, "w") as f:
        json.dump(new_data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuscenes_dir', default="/workspace/nuscenes/")
    parser.add_argument('--json_dir', default="/workspace/blip2_mod/docs/_static/nuscenes_test.json")
    args = parser.parse_args()
    exit(main(args.nuscenes_dir, args.json_dir))