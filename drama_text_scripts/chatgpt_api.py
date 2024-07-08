########## GPT setting code ##########
import openai
import pdb
import json
from tqdm import tqdm
import argparse
import math
import time

def main(nuscenes_dir: str, json_dir: str):
    My_OpenAI_key = ''
    openai.api_key = My_OpenAI_key

    chat_completion = openai.ChatCompletion()
    temperature = 0.5
    max_tokens = 2000

    ask_zero = """
    Your Role: You are a writer tasked with generating descriptive captions about objects without including their location information.

    Inputs Explained:
    1. Caption: Describes an object but might contain location info which we don't want.
    2. GT class: The actual type of the object.
    3. Maneuver: Predicted future movement of the object.

    Your Task:
    - Create a single caption.
    - DO NOT include location information like 'left side', 'right lane', 'away from the ego car', 'in the ego lane' etc.
    - If the object described in the Caption is different from the GT class, craft your caption using only the GT class and Maneuver.
    - If the action described in the Caption does not align with the Maneuver, adjust the description to fit the provided Maneuver.
    - Explicitly mention the object's expected maneuver using the provided "Maneuver" input.

    Example Input 1:
    Caption: a white suv driving in the left lane, away from the intersection, in the rain
    GT class: vehicle.car
    Maneuver: straight

    Bad Output: A white SUV is driving in the left lane, away from the intersection, in the rain, and is expected to continue straight.
    Good Output: A white SUV is driving in the rain, and is expected to continue straight.

    Example Input 2:
    Caption: a white suv parked on the left side of the road
    GT class: vehicle.car
    Maneuver: stationary

    Bad Output: A white SUV is parked on the left side of the road.
    Good Output: A white SUV is parked, and is anticipated remain stationary.

    Now, based on the above guidelines and examples, create a caption for the following:
    """
    # print(ask_zero)
    
    messages = [{'role': "user", "content": ask_zero}]
    response = chat_completion.create(
                            messages=messages, 
                            model='gpt-3.5-turbo',
                            temperature=temperature,
                            max_tokens=max_tokens,
                            )
    answer = response.choices[0].message['content'].strip()
    print(answer)

    start_time = time.time()
    with open(json_dir, "r") as f:
        data = json.load(f)
    
    new_data = {}
    image_count = 0
    for image_path, object_list in tqdm(data.items(), desc="Processing images", position=0):
        new_obj_list = []
        for obj in tqdm(object_list, desc=f"Processing objects in an image", position=1, leave=False):
            if "caption" in obj and "maneuver" in obj and "chatgpt_caption" not in obj and "fake_caption" not in obj:
                GT_class = obj["category_name"]
                GT_maneuver = obj["maneuver"]
                caption = obj["caption"]
                prompt = f"""(Caption: {caption}\nGT class: {GT_class}\nmaneuver: {GT_maneuver}\n)\n"""
            
                messages.append({'role': "assistant", "content": prompt})
                max_retries = 10
                for _ in range(max_retries):
                    try:
                        response = chat_completion.create(
                                            messages=messages, 
                                            model='gpt-3.5-turbo',
                                            max_tokens=max_tokens,    
                                            temperature=temperature,
                                            )
                        answer = response.choices[0].message['content'].strip()
                        obj["chatgpt_caption"] = answer
                        break
                    except openai.error.ServiceUnavailableError:
                        print("Service is unavailable. Retrying...")
                        time.sleep(5)
                    except Exception as e:
                        print("An unexpected error occurred:", str(e))
                        time.sleep(10)

                messages.pop()
                time.sleep(0.1)

            new_obj_list.append(obj) 
        
        new_data[image_path] = new_obj_list
        image_count += 1
        if image_count != 0 and image_count % 10000 == 0:
            with open(f"/workspace/nuScenes/v1.0-trainval/final_professor_{image_count}.json", "w") as f:
                json.dump(new_data, f, indent=4)
            new_data = {}
            
    if new_data:
        with open(f"/workspace/nuScenes/v1.0-trainval/final_professor_{image_count}.json", "w") as f:
            json.dump(new_data, f, indent=4)
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuscenes_dir', default="/workspace/nuScenes/")
    parser.add_argument('--json_dir', default="/workspace/nuScenes/v1.0-trainval/rest.json")
    args = parser.parse_args()
    exit(main(args.nuscenes_dir, args.json_dir))