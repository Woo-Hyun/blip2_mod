import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import os

import logging
from packaging import version

from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
import transformers

from lavis.models.blip2_models.blip2_opt import Blip2OPT
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.datasets.datasets.caption_datasets import convert_geometry_to_bbox, concatenate_images


def main():
    # setup device to use
    device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
    # load sample image
    image = Image.open("./docs/_static/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883536912466.jpg").convert("RGB")
    save_path = os.path.join("/workspace/LAVIS/image_test/", f"image.png")
    image.save(save_path)
    cropped_image = image.crop((768.5951699939305,
                434.7182566728917,
                838.9851232415795,
                514.4538944717932))
    

    save_path = os.path.join("/workspace/LAVIS/image_test/", f"crop_image.png")
    cropped_image.save(save_path)

    image = concatenate_images(image, cropped_image)
    save_path = os.path.join("/workspace/LAVIS/image_test/", f"concate_image.png")
    image.save(save_path)
    
    # loads BLIP-2 pre-trained model
    # _, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)
    # print(type(model))

    # model = Blip2OPT()
    # Load the fine-tuned weights
    fine_tuned_weights_path = "/workspace/LAVIS/lavis/output/BLIP2/Caption_coco/20230821065/checkpoint_best.pth"
    # fine_tuned_weights_path = "/workspace/LAVIS/lavis/output/BLIP2/Caption_coco/20230828105/checkpoint_best.pth"
    checkpoint = torch.load(fine_tuned_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    # model.float()
    model.to(device)
    # model.load_state_dict(torch.load(fine_tuned_weights_path, map_location=device))
    # model = torch.load(fine_tuned_weights_path, map_location=device)['model']
    # model.eval()

    image = vis_processors["eval"](image).unsqueeze(0).to(device)

    text = model.generate({"image": image})

    print(text)


if __name__ == "__main__":
    main()