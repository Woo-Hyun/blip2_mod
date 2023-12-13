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

NameMappingVehicle = {
        'vehicle.bicycle',
        'vehicle.bus.bendy',
        'vehicle.bus.rigid',
        'vehicle.car',
        'vehicle.construction',
        'vehicle.motorcycle',
        'vehicle.trailer',
        'vehicle.truck'
}

def process_maneuver(traj:np.array, traj_mask:np.array, category:str
                        ) -> str :
    # classifier for maneuver 
    # kMaxDisplacementForStationary = 3.0          # (m)
    # kMaxLateralDisplacementForStraight = 3.0     # (m)
    # kMinLongitudinalDisplacementForUTurn = -5.0  # (m)
    # kMaxAbsHeadingDiffForStraight = 0.35   # (rad)
    # heading_delta_threshold = 2.85

    vehicle_displacement_for_stationary = 3.0          # (m)
    pedestrian_displacement_for_stationary = 1.5          # (m)
    vehicle_lateral_displacement_for_straight = 3.0     # (m)
    pedestrian_lateral_displacement_for_straight = 2.0     # (m)
    heading_delta_for_straight = 0.35   # (rad) 20 degree
    lane_change_threshold = 1.5
    vehicle_longitudinal_displacement_for_uturn = -5.0  # (m)
    pedestrian_logitudinal_displacement_for_uturn = -1.5  # (m)


    agent_positions = torch.tensor(traj)
    origin_heading_vector = agent_positions[0]
    origin_heading_delta = torch.atan2(origin_heading_vector[1], origin_heading_vector[0])
    origin_rotate_mat = torch.tensor([[torch.cos(origin_heading_delta), -torch.sin(origin_heading_delta)],
                            [torch.sin(origin_heading_delta), torch.cos(origin_heading_delta)]])
    
    agent_positions = torch.matmul(agent_positions,origin_rotate_mat)

    if np.any(traj_mask[:,0] == 1):
        last_valid_index = np.max(np.where(traj_mask[:,0] == 1))
    else:
        last_valid_index = 0
    start_heading_vector = agent_positions[0]
    start_heading_delta = torch.atan2(start_heading_vector[1], start_heading_vector[0])
    final_heading_vector = agent_positions[last_valid_index] - agent_positions[last_valid_index - 1]
    final_heading_delta = torch.atan2(final_heading_vector[1], final_heading_vector[0]) 
    # if np.abs(final_heading_delta) > heading_delta_threshold :
    #     final_heading_delta = 0

    heading_delta = final_heading_delta - start_heading_delta
    xy_delta = agent_positions[last_valid_index]
    final_displacement = np.linalg.norm(xy_delta)

    # for vehicle
    if category in NameMappingVehicle:
        if final_displacement < vehicle_displacement_for_stationary:
            return "stationary"
        if (np.abs(xy_delta[1]) < vehicle_lateral_displacement_for_straight) and (np.abs(heading_delta) < heading_delta_for_straight):
            if (np.abs(xy_delta[1]) > lane_change_threshold):
                return "lane change"
            return "straight"
        if heading_delta < -heading_delta_for_straight and xy_delta[1]:
            if xy_delta[0] < vehicle_longitudinal_displacement_for_uturn:
                return "right U-turn"
            else:
                return "right turn"
        if xy_delta[0] < vehicle_longitudinal_displacement_for_uturn:
            return "left U-turn"
        return "left turn"
    
    
    # for pedestrian
    else:
        if final_displacement < pedestrian_displacement_for_stationary:
            return "stationary"
        if (np.abs(xy_delta[1]) < pedestrian_lateral_displacement_for_straight) and (np.abs(heading_delta) < heading_delta_for_straight):
            return "straight"
        if heading_delta < -heading_delta_for_straight and xy_delta[1]:
            if xy_delta[0] < pedestrian_logitudinal_displacement_for_uturn:
                return "right U-turn"
            else:
                return "right turn"
        if xy_delta[0] < pedestrian_logitudinal_displacement_for_uturn:
            return "left U-turn"
        return "left turn"

    # if final_displacement < kMaxDisplacementForStationary:
    #     return "stationary"
    # if (np.abs(xy_delta[1]) < kMaxLateralDisplacementForStraight) or (np.abs(heading_delta) < kMaxAbsHeadingDiffForStraight):
    #     if (category in NameMappingManeuver) and (np.abs(xy_delta[1]) > lane_change_threshold):
    #         return "lane_change"
    #     return "straight"
    # if heading_delta < -kMaxAbsHeadingDiffForStraight and xy_delta[1]:
    #     return "right_u_turn" if xy_delta[0] < kMinLongitudinalDisplacementForUTurn \
    #         else "right_turn"
    # if xy_delta[0] < kMinLongitudinalDisplacementForUTurn:
    #     return "left_u_turn"
    # return "left_turn"

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

def main(nuscenes_dir: str, json_dir: str):
    ##### set NuScenes #####
    nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_dir, verbose=True)
    nusc_traj = NuScenesTraj(nusc,
                            predict_steps=12,
                            past_steps=4)
    ########################

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

            if category in NameMapping:
                # test_sample = nusc.sample_data(obj["sample_data_token"])
                sample_data = nusc.get('sample_data', obj["sample_data_token"])
                sample_token = sample_data['sample_token']
                sample = nusc.get('sample', sample_token)

                ann_tokens = np.array(sample['anns'])
                # import pdb; pdb.set_trace()
                # anno_info = nusc.get('sample_annotation', obj['sample_annotation_token'])
                # anno_info['rotation']
                gt_fut_traj, gt_fut_traj_mask, gt_past_traj, gt_past_traj_mask, bev_pos, distance,agent_type,vel,acc,heading,orient_pose = nusc_traj.get_traj_label_single_instance(
                    sample['token'], ann_tokens, obj['instance_token'])
                gt_past_traj = np.flipud(gt_past_traj[0])
                gt_past_traj_mask = np.flipud(gt_past_traj_mask[0])
                gt_fut_traj = gt_fut_traj[0]
                gt_fut_traj_mask = gt_fut_traj_mask[0]
                gt_traj = np.concatenate((gt_past_traj, gt_fut_traj), axis=0)
                gt_traj_mask = np.concatenate((gt_past_traj_mask, gt_fut_traj_mask), axis=0)
                
                maneuver = process_maneuver(gt_fut_traj, gt_fut_traj_mask, category)
                # obj["maneuver"] = maneuver
                # obj["bev_pos"] = [bev_pos[0][0][0], bev_pos[0][0][1]]
                
                past_array = gt_past_traj
                past_array_mask = gt_past_traj_mask
                past_traj = past_array.tolist()
                past_traj_mask = past_array_mask.tolist()
                past_traj.pop()
                past_traj_mask.pop()
                obj["past_traj"] = past_traj
                obj["past_traj_mask"] = past_traj_mask
                # pdb.set_trace()

            new_obj_list.append(obj)
        
        new_data[image_path] = new_obj_list

    # new_json_dir = "/workspace/nuScenes/v1.0-trainval/nuscenes_test_past_traj.json"
    new_json_dir = "/workspace/blip2_mod/docs/_static/nuscenes_test_past_traj.json"
    with open(new_json_dir, "w") as f:
        json.dump(new_data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuscenes_dir', default="/workspace/nuscenes/")
    # parser.add_argument('--json_dir', default="/workspace/nuScenes/v1.0-trainval/image_annotations_with_caption_filtered.json")
    parser.add_argument('--json_dir', default="/workspace/blip2_mod/docs/_static/maneuver_fake_filtered.json")
    args = parser.parse_args()
    exit(main(args.nuscenes_dir, args.json_dir))