import torch
from PIL import Image, ImageDraw
from lavis.models import load_model_and_preprocess
import os
import json
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pdb

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

NameMapping = {
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
}

def main(nuscenes_dir: str, json_dir: str):

    with open(json_dir, "r") as f:
        data = json.load(f)

    new_data = {}
    for image_path, object_list in tqdm(data.items(), desc="Processing images", position=0):
        image = Image.open(os.path.join(nuscenes_dir, image_path)).convert("RGB")
        
        new_obj_list = []
        for obj in tqdm(object_list, desc=f"Processing objects in {image_path}", position=1, leave=False):
            category = obj["category_name"]

            if "maneuver" in obj and "caption" not in obj:
                if obj["maneuver"] == "stationary":
                    obj["fake_caption"] = "The " + category + " will be stationary."
                elif obj["maneuver"] == "lane change":
                    obj["fake_caption"] = "The " + category + " will change lane."
                elif obj["maneuver"] == "straight":
                    obj["fake_caption"] = "The " + category + " will go straight."
                elif obj["maneuver"] == "left turn":
                    obj["fake_caption"] = "The " + category + " will turn left."
                elif obj["maneuver"] == "right turn":
                    obj["fake_caption"] = "The " + category + " will turn right."
                elif obj["maneuver"] == "left U-turn":
                    obj["fake_caption"] = "The " + category + " will make a left U-turn."
                else:
                    obj["fake_caption"] = "The " + category + " will make a right U-turn."
                # pdb.set_trace()

            new_obj_list.append(obj)
        
        new_data[image_path] = new_obj_list

    new_json_dir = "/workspace/nuScenes/v1.0-trainval/image_annotations_fake_caption.json"
    # new_json_dir = "/workspace/blip2_mod/docs/_static/maneuver_fake.json"
    with open(new_json_dir, "w") as f:
        json.dump(new_data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuscenes_dir', default="/workspace/nuScenes/")
    parser.add_argument('--json_dir', default="/workspace/nuScenes/v1.0-trainval/image_annotations_with_caption_maneuver.json")
    args = parser.parse_args()
    exit(main(args.nuscenes_dir, args.json_dir))