"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import torch
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image

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


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
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

        return {
            "image": image,
            "text_input": caption,
            # "image_id": self.img_ids[ann["image_id"]],
        }


class CaptionEvalDataset(BaseDataset, __DisplMixin):
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

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
