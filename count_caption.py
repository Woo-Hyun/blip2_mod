import json

# JSON 파일 경로
json_file_path = '/workspace/nuScenes/v1.0-trainval/image_annotations_with_caption.json'

# "caption" attribute가 포함된 오브젝트의 수를 세기 위한 카운터
caption_count = 0

# JSON 파일 열기
with open(json_file_path, 'r') as f:
    data = json.load(f)

# JSON 파일 내용 순회
for image_path, object_list in data.items():
    for obj in object_list:
        if "caption" in obj:
            caption_count += 1

# 결과 출력
print(f'The number of objects with "caption" attribute is: {caption_count}')