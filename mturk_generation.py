import torch
from PIL import Image, ImageDraw
from lavis.models import load_model_and_preprocess
import os
import json
import argparse
from tqdm import tqdm
import time
import random

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
    else:
        if int(crop_width * (orig_height / crop_height)) == 0 or orig_height == 0:
            return original
        cropped_resized = cropped.resize((int(crop_width * (orig_height / crop_height)), orig_height))
    
    return original, cropped_resized

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
    image_json_dir = "/workspace/blip2_mod/mturk/single/image/"
    crop_json_dir = "/workspace/blip2_mod/mturk/single/crop/"
    text_json_dir = "/workspace/blip2_mod/mturk/single/text/"
    new_json_dir = "/workspace/blip2_mod/mturk/single/mturk.json"

    with open(json_dir, "r") as f:
        data = json.load(f)

    # 'chatgpt_caption'이 있는 모든 이미지를 필터링
    filtered_keys = [key for key in data if any("chatgpt_caption" in obj and obj["chatgpt_caption"].strip() for obj in data[key])]
    
    new_data = {}
    idx = 0
    while idx < 2000 and filtered_keys:
        image_path = random.choice(filtered_keys)  # 무작위 이미지 선택
        filtered_keys.remove(image_path)  # 선택된 이미지는 제거

        image = Image.open(os.path.join(nuscenes_dir, image_path)).convert("RGB")
        object_list = data[image_path]

        max_size = 0
        selected_obj = None
        image_area = image.size[0] * image.size[1]  # 전체 이미지 크기 계산

        # 오브젝트 리스트를 순회하면서 조건을 만족하는 가장 큰 오브젝트를 찾음
        for obj in object_list:
            if "chatgpt_caption" in obj and obj["chatgpt_caption"].strip():
                x_min, y_min, x_max, y_max = obj["bbox_corners"]
                bbox_area = (x_max - x_min) * (y_max - y_min)  # 바운딩 박스 크기 계산
                if bbox_area < 0.7 * image_area and bbox_area > max_size:  # 바운딩 박스 크기가 이미지 크기의 50% 미만인지 확인
                    max_size = bbox_area
                    selected_obj = obj

        # 조건을 만족하는 오브젝트를 찾았다면 해당 오브젝트에 대해 처리
        if selected_obj:
            cropped_image = image.crop(tuple(selected_obj["bbox_corners"]))
            original, cropped = concatenate_images(image, cropped_image)
            
            file_index = format(idx, '04')  # 0000, 0001, ..., 1999 형태로 인덱스 포매팅
            image_save_path = os.path.join(image_json_dir, f"{file_index}.jpg")
            crop_save_path = os.path.join(crop_json_dir, f"{file_index}.jpg")
            text_save_path = os.path.join(text_json_dir, f"{file_index}.txt")
            original.save(image_save_path)
            cropped.save(crop_save_path)
            with open(text_save_path, "w") as text_file:
                text_file.write(selected_obj["chatgpt_caption"])

            new_data[image_path] = [selected_obj]
            idx += 1  # 인덱스 증가

    # JSON 파일 저장
    with open(new_json_dir, "w") as f:
        json.dump(new_data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuscenes_dir', default="/workspace/nuscenes_bp/data/nuscenes")
    # parser.add_argument('--json_dir', default="/workspace/nuScenes/v1.0-trainval/image_annotations_arrange.json")
    parser.add_argument('--json_dir', default="/workspace/nuscenes_bp/data/nuscenes/v1.0-trainval/with_text_v1.0/final_merged.json")
    args = parser.parse_args()
    exit(main(args.nuscenes_dir, args.json_dir))