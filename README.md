# nuScenes-Text Dataset

This repository contains the nuScenes-Text dataset for autonomous driving research. This dataset is based on [nuScenes](https://www.nuscenes.org/) and includes text descriptions of objects captured by the cameras in the nuScenes dataset.

## Dataset Structure

The dataset is organized into the following structure:

{ 
  "image_file_name": [                    
    {
        "attribute_tokens": 
        "bbox_corners": 
        "category_name": 
        "filename": 
        "instance_token": 
        "next":
        "num_lidar_pts": 
        "num_radar_pts": 
        "prev":
        "sample_annotation_token": 
        "sample_data_token": 
        "visibility_token": 
        "caption": 
        "maneuver":
        "bev_pos":
        "chatgpt_caption": 
        "chatgpt_caption_2": 
        "chatgpt_caption_3": 
        "fake_caption": 
    },
    another objects in this image ...
  ],
  "other_image_file_name": ...
}