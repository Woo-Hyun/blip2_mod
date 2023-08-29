"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset

COCOCapDataset = CaptionDataset

def convert_geometry_to_bbox(geometry):
    """
    Convert the geometry format to bbox format.

    Args:
    - geometry (list): A list of four coordinate points.

    Returns:
    - list: A bounding box in [x, y, width, height] format.
    """
    x_coords = [point[0] for point in geometry]
    y_coords = [point[1] for point in geometry]
    
    x = min(x_coords)
    y = min(y_coords)
    width = max(x_coords) - x
    height = max(y_coords) - y
    
    return [x, y, width, height]

def concatenate_images(original, cropped):
    crop_width, crop_height = cropped.size
    orig_width, orig_height = original.size
    
    if crop_height < crop_width:
        cropped_resized = cropped.resize((orig_width, int(crop_height * (orig_width / crop_width))))
        
        concatenated = Image.new('RGB', (orig_width, orig_height + cropped_resized.height))
        concatenated.paste(original, (0, 0))
        concatenated.paste(cropped_resized, (0, orig_height))
    else:
        cropped_resized = cropped.resize((int(crop_width * (orig_height / crop_height)), orig_height))
        
        concatenated = Image.new('RGB', (orig_width + cropped_resized.width, orig_height))
        concatenated.paste(original, (0, 0))
        concatenated.paste(cropped_resized, (orig_width, 0))
    
    return concatenated



class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        geometry = ann["geometry"]
        left = min([point[0] for point in geometry])
        upper = min([point[1] for point in geometry])
        right = max([point[0] for point in geometry])
        lower = max([point[1] for point in geometry])
        cropped_image = image.crop((left, upper, right, lower))

        image = concatenate_images(image, cropped_image)

        # save_path = os.path.join("/workspace/LAVIS/image_test/", f"image_{index}.png")
        # image.save(save_path)
        # print("saved")

        image = self.vis_processor(image)
        # cropped_image = self.vis_processor(cropped_image)
        caption = self.text_processor(ann["caption"])

        # concatenate image and cropped image
        #image = torch.cat((image, cropped_image), dim=0)

        img_id = ann["image"].split("/")[-1].strip(".png").split("_")[-1]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }


class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["img_id"]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }
