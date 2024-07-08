import torch
from PIL import Image, ImageDraw
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


def main():
    # setup device to use
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    # load sample image
    image = Image.open("./docs/_static/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915290412465.jpg").convert("RGB") # 1600 by 900
    save_path = os.path.join("/workspace/blip2_mod/image_test/", f"image.png")
    image.save(save_path)
    cropped_image = image.crop((1257.2958175561353,
                420.10264293214874,
                1333.5614778900074,
                540.542018436124))
    
    save_path = os.path.join("/workspace/blip2_mod/image_test/", f"crop_image.png")
    cropped_image.save(save_path)

    image = concatenate_images(image, cropped_image)
    save_path = os.path.join("/workspace/blip2_mod/image_test/", f"concate_image.png")
    image.save(save_path)
    
    ###############################
    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)
    # print(type(model))

    # Load the fine-tuned weights
    fine_tuned_weights_path = "/workspace/blip2_mod/lavis/output/BLIP2/Caption_coco/20230821065_opt/checkpoint_best.pth"
    # fine_tuned_weights_path = "/workspace/blip2_mod/lavis/output/BLIP2/Caption_coco/20230906090_few/checkpoint_best.pth"
    checkpoint = torch.load(fine_tuned_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    ###############################

    image = vis_processors["eval"](image).unsqueeze(0).to(device)

    text = model.generate({"image": image}, temperature=0.5)
    # sec_text = model.generate({"image": image, "prompt": "Questions: What is the observed object doing now? Answer:"})
    # thd_text = model.generate({"image": image, "prompt": "Questions: How should I move to prevent collisions? Answer:"})
    # text = model.generate({"image": image})

    print(text)


if __name__ == "__main__":
    main()