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

with open("/workspace/nuScenes/v1.0-trainval/image_annotations_with_caption_maneuver_fakecaption.json", "r") as f:
    data = json.load(f)

new_data = {}
for image_path, object_list in tqdm(data.items(), desc="Processing images", position=0):
        
    new_obj_list = []
    for obj in tqdm(object_list, desc=f"Processing objects in {image_path}", position=1, leave=False):
        if "fake_caption" in obj:
            for key, value in NameMapping.items():
                obj["fake_caption"] = obj["fake_caption"].replace(key, value)
        new_obj_list.append(obj)
        
    new_data[image_path] = new_obj_list

new_json_dir = "/workspace/nuScenes/v1.0-trainval/image_annotations.json"
with open(new_json_dir, "w") as f:
    json.dump(new_data, f, indent=4)