########## GPT setting code ##########
import openai
import pdb
import json
from tqdm import tqdm
import argparse
import math
import time

def main(nuscenes_dir: str, json_dir: str):
    My_OpenAI_key = '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'
    openai.api_key = My_OpenAI_key

    chat_completion = openai.ChatCompletion()
    temperature = 0.5
    max_tokens = 2000

    # for merged caption with location data in one image
    # ask_zero = """I am now working on the vehicle prediction of self-driving cars.
    #             To increase performance, we use ego central camera view images to generate text and try to use that text.
    #             The pre-trained model was fine-tuned on a custom dataset and used that model to create a capture of objects corresponding to 2d Bboxes was created.
    #             It also created future movements through data processing.
    #             You are a machine box that fuses some datas well and make it into a single text, if I provide the generated caption, GT maneuver, and GT class of each object.
    #             In particular, I will provide data on multiple objects that exist in one image.
    #             So please understand the relationship between the data that exist in one image and create a caption for each object.
    #             Here are some rules for you.
    #             Rule 1: Don't mention GT class but check if it fits well with the original caption and create a caption to follow GT class if it's wrong.
    #             Rule 2: When creating fused captions, make them without the object's location text like 'left', 'right', 'left to right', 'right to left' that exists in the provided original captions.
                # Rule 2: Remove any location-specific text from the caption like 'left side of the road', 'in right lane', 'from left to right', 'from right to left'.
    #             Rule 3: Create captions to match the number of objects I provided. If you provided three object, you should create three captions. If you provided four object, you should create four captions.
    #             Rule 4: Don't make merged caption about whole image.
                # Rule 1: Check the caption against the GT class. If the caption doesn't match the GT class, adjust it without mentioning the GT class explicitly.
    #             I'll give you two output examples that is expected from you, each of which is about an object and corresponding caption that exist in one image have two objects.
    #             object 1: The white SUV is parked and is anticipated to stay stationary.
    #             object 2: A construction worker is expected to remain standing still.
    #             I will give you list of brackets, each bracket contains about one object.
    #             Please format the result in the order of provided."""

    # for caption for each object
    # ask_zero = """I am now working on the vehicle prediction of self-driving cars.
    #             To increase performance, we use ego central camera view images to generate text and try to use that text.
    #             The pre-trained model was fine-tuned on a custom dataset and used that model to create a capture of objects corresponding to 2d Bboxes was created.
    #             It also created future movements that called maneuver through data processing.
    #             You are a machine box that fuses some datas well and make it into a single text, if I provide the generated caption, GT maneuver, and GT class of an object.
    #             I'll provide you with the rules you need to follow when you create caption.
    #             Rule 1: Remove any location-specific text from the caption like 'left side of the road', 'in right lane', 'from left to right', 'from right to left', 'away from the ego car'.
    #             Rule 2: Incorporate the GT maneuver into the caption to indicate expected future movement.
    #             I'll give you an input example for you, and output example that is expected from you.
    #             input:
    #             (GT class: human.pedestrian.adult
    #             caption: a pedestrian wearing a white shirt and black pants, walking on the right side of the road, away from the ego car
    #             GT maneuver: right_turn)
    #             expected output:
    #             fused caption: a pedestrian wearing a white shirt and black pants, and is anticipated to turn right."""

    ask_zero = """
    Your Role: You are a writer tasked with generating descriptive captions about objects without including their location information.

    Inputs Explained:
    1. Target object: The object for which we want to generate a caption.
    2. Other objects: Other objects in the image that might interact with the target object.
    3. Caption: Describes an object but might contain location info which we don't want.

    Your Task:
    - Create a single caption for the target object.
    - DO NOT include location information like 'left side', 'right lane', 'away from the ego car', 'in the ego lane' etc.
    - Consider the interactions between the target object and other objects in the image, including their BEV positions.
    - If the object described in the Caption is different from the GT class of the target object, craft your caption using only the GT class and Maneuver of the target object.
    - If the action described in the Caption does not align with the Maneuver of the target object, adjust the description to fit the provided Maneuver.
    - Explicitly mention the target object's expected maneuver using the provided "Maneuver" input.

    Example Input:
    Target object:
    Category: vehicle.car
    Maneuver: stationary
    BEV position:
    bev_x: -3.5543665287018373
    bev_y: 8.18180684176209
    Caption: A silver SUV stopped in the middle of the intersection because of a red traffic light.

    Other objects:
    Category: human.pedestrian.adult
    Maneuver: right turn
    BEV position:
    bev_x: -3.472824141395617
    bev_y: 16.291993451148567

    Category: vehicle.car
    Maneuver: left turn
    BEV position:
    bev_x: 0.18532427848334937
    bev_y: 32.95069544554596

    Category: vehicle.car
    Maneuver: straight
    BEV position:
    bev_x: 14.257511027186142
    bev_y: 34.74069128621959

    Bad Output: A silver SUV stopped in the middle of the intersection because of a red traffic light. There is a pedestrian nearby who is going to turn right, and other cars are turning left and going straight.

    Good Output: A silver SUV is stationary in the middle of the intersection because of a red traffic light and a pedestrian preparing to turn right. Meanwhile, another car is making a left turn, and a third car is moving straight ahead.

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
    # answer
    # messages.append({'role': "assistant", "content":answer})
    # # request
    # messages.append({'role': "user", "content":"""(GT class: vehicle.car
    #                                                 caption: a white sedan parked on the left side of the intersection
    #                                                 GT maneuver: right_turn)"""})
    # messages.append({'role': "assistant", "content": """A white sedan is parked on the intersection and is anticipated to make a right turn."""})
    # # request
    # messages.append({'role': "user", "content":"""(GT class: vehicle.car
    #                                                 caption: a white suv driving in the left lane, away from the intersection, in the rain
    #                                                 GT maneuver: straight)"""})
    # messages.append({'role': "assistant", "content": """A white SUV is driving, and is expected to continue straight, in the rain."""})

    start_time = time.time()
    with open(json_dir, "r") as f:
        data = json.load(f)
    
    new_data = {}
    image_count = 0
    for image_path, object_list in tqdm(data.items(), desc="Processing images", position=0):
        new_obj_list = []
        for obj in tqdm(object_list, desc=f"Processing objects in an image", position=1, leave=False):
            if "caption" in obj and "maneuver" in obj and "fake_caption" not in obj:
                # 대상 오브젝트의 데이터를 문자열로 변환
                target_str = ""
                category_name = obj['category_name']
                maneuver = obj['maneuver']
                # past_traj = obj['past_traj']
                bev_pos = obj['bev_pos']
                caption = obj['caption']
                # x_coords = ', '.join(map(str, [pt[0] for pt in past_traj]))
                # y_coords = ', '.join(map(str, [pt[1] for pt in past_traj]))
                bev_x, bev_y = map(str, bev_pos)
                target_str += f"Category: {category_name}\nManeuver: {maneuver}\nBEV pos:\n  bev_x: {bev_x}\n  bev_y: {bev_y}\ncaption: {caption}\n"
                
                # 이미지에 있는 다른 객체의 데이터를 문자열로 변환
                other_objects_str = ""
                for other_obj in object_list:
                    if other_obj is not obj and "caption" in other_obj and "maneuver" in other_obj and "fake_caption" not in other_obj:
                        category_name = other_obj['category_name']
                        maneuver = other_obj['maneuver']
                        # past_traj = other_obj['past_traj']
                        bev_pos = other_obj['bev_pos']
                        caption = other_obj['caption']
                        # x_coords = ', '.join(map(str, [pt[0] for pt in past_traj]))
                        # y_coords = ', '.join(map(str, [pt[1] for pt in past_traj]))
                        bev_x, bev_y = map(str, bev_pos)
                        other_objects_str += f"Category: {category_name}\nManeuver: {maneuver}\nBEV pos:\n  bev_x: {bev_x}\n  bev_y: {bev_y}\n"
                
                # 프롬프트 구성
                prompt = f"Target object:\n{target_str}Other objects:\n{other_objects_str}"
                
                messages.append({'role': "assistant", "content": prompt})
                max_retries = 10  # 최대 재시도 횟수
                for _ in range(max_retries):
                    try:
                        response = chat_completion.create(
                                            messages=messages, 
                                            model='gpt-3.5-turbo',
                                            max_tokens=max_tokens,    
                                            temperature=temperature,
                                            )
                        answer = response.choices[0].message['content'].strip()
                        obj["chatgpt_interaction_caption"] = answer
                        break  # 요청이 성공하면 재시도 루프를 빠져나옵니다.
                    except openai.error.ServiceUnavailableError:
                        print("Service is unavailable. Retrying...")
                        time.sleep(5)  # 잠시 대기하고 다시 시도합니다.
                    except Exception as e:
                        print("An unexpected error occurred:", str(e))
                        time.sleep(10) # 다른 예외가 발생하면 다시 시도

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
        # new_json_dir = "/workspace/nuScenes/v1.0-trainval/image_annotations_with_caption_maneuver.json"
        # new_json_dir = "/workspace/blip2_mod/docs/_static/nuscenes_test_few.json"
        with open(f"/workspace/blip2_mod/docs/_static/nuscenes_test_chatgpt_interaction.json", "w") as f:
            json.dump(new_data, f, indent=4)
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuscenes_dir', default="/workspace/nuScenes/")
    # parser.add_argument('--json_dir', default="/workspace/nuScenes/v1.0-trainval/rest_professor.json")
    parser.add_argument('--json_dir', default="/workspace/blip2_mod/docs/_static/nuscenes_test_chatgpt.json")
    args = parser.parse_args()
    exit(main(args.nuscenes_dir, args.json_dir))