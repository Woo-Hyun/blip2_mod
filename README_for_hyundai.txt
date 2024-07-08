This document is related to the creation of the text dataset used in V-Trap.

The text generation process is divided into three stages as follows:
1. Training of VLM (blip2)
2. Primary text generation through the trained VLM
3. Secondary text generation via LLM (chatgpt)

1. Training of VLM (blip2) -------------------------------------------------------------------------
1-1. For the training environment setup of blip2, refer to the existing blip2 environment setup README.
1-2. Move the files located in the enclosed drama_caption_data path to the same location as the existing drama dataset.
     These data have been modified from the original drama data format to be suitable for training.
1-3. Running /run_scripts/blip2/train/train_caption_coco.sh will initiate the training process.
     # bash /run_scripts/blip2/train/train_caption_coco.sh 
     Depending on the computer environment, it may be necessary to appropriately modify the /blip2_mod/lavis/projects/blip2/train/caption_coco_ft.yaml file or move the files used for training.
1-4. The trained model can be found in /lavis/output.
----------------------------------------------------------------------------------------------------


2. Primary text generation through the trained VLM -------------------------------------------------
2-1. To generate the primary text using the trained model, run mod_nuscenes_caption.py.
     The input file name and the name of the saved file need to be appropriately changed.
----------------------------------------------------------------------------------------------------


3. Secondary text generation via LLM (chatgpt) -----------------------------------------------------
3-1. Run mod_nuscenes_maneuver.py to add maneuvers.
3-2. To generate secondary text using the chatgpt API, run chatgpt_api.py.
     The API key must be newly entered.
3-3. The final generated dataset is enclosed with the name nuscenes_with_text.json.
----------------------------------------------------------------------------------------------------