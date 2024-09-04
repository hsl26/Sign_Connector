import json
import torch
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# 디렉토리 내의 모든 JSON 파일 경로 읽기
def get_json_files_from_directory(directory):
    json_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    return sorted(json_files)

directory_path = "keypoints/01/NIA_SL_WORD0001_REAL01_F"

path_list = get_json_files_from_directory(directory_path)

print(len(get_json_files_from_directory(directory_path)))

def load_json_files(json_files):
    data_dicts = []
    for file_path in json_files:
        with open(file_path, 'r') as file:
            data_dict = json.load(file)
            data_dicts.append(data_dict)
    return data_dicts

data_dic_list = load_json_files(path_list)
print(len(load_json_files(path_list)))

def extract_and_combine_keypoints(data_dicts):
    combined_keypoints = []

    for data_dict in data_dicts:
        people = data_dict.get('people', {})

        # 2D 키포인트
        pose_keypoints_2d = people.get('pose_keypoints_2d', [])
        face_keypoints_2d = people.get('face_keypoints_2d', [])
        hand_left_keypoints_2d = people.get('hand_left_keypoints_2d', [])
        hand_right_keypoints_2d = people.get('hand_right_keypoints_2d', [])

        # 3D 키포인트
        pose_keypoints_3d = people.get('pose_keypoints_3d', [])
        face_keypoints_3d = people.get('face_keypoints_3d', [])
        hand_left_keypoints_3d = people.get('hand_left_keypoints_3d', [])
        hand_right_keypoints_3d = people.get('hand_right_keypoints_3d', [])

        def convert_to_tensor(keypoints, dim):
            if len(keypoints) % dim != 0:
                padding_size = dim - (len(keypoints) % dim)
                keypoints += [0] * padding_size
            return torch.tensor(keypoints, dtype=torch.float32).view(-1, dim)

        # 각 특징점을 텐서로 변환하고, 2D 형식으로 변경
        pose_tensor_2d = convert_to_tensor(pose_keypoints_2d, 4)
        face_tensor_2d = convert_to_tensor(face_keypoints_2d, 4)
        hand_left_tensor_2d = convert_to_tensor(hand_left_keypoints_2d, 4)
        hand_right_tensor_2d = convert_to_tensor(hand_right_keypoints_2d, 4)
        pose_tensor_3d = convert_to_tensor(pose_keypoints_3d, 4)
        face_tensor_3d = convert_to_tensor(face_keypoints_3d, 4)
        hand_left_tensor_3d = convert_to_tensor(hand_left_keypoints_3d, 4)
        hand_right_tensor_3d = convert_to_tensor(hand_right_keypoints_3d, 4)

        # 텐서 결합 (N, 3) 형식의 텐서들을 하나의 텐서로 결합
        combined_tensor_2d = torch.cat((pose_tensor_2d, face_tensor_2d, hand_left_tensor_2d, hand_right_tensor_2d), dim=0)
        combined_tensor_3d = torch.cat((pose_tensor_3d, face_tensor_3d, hand_left_tensor_3d, hand_right_tensor_3d), dim=0)

        # 2D와 3D 데이터를 최종 결합
        combined_tensor = torch.cat((combined_tensor_2d, combined_tensor_3d), dim=0)

        combined_keypoints.append(combined_tensor)
    return combined_keypoints

print(len(extract_and_combine_keypoints(data_dic_list)))
