import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import os
import json
from sklearn.preprocessing import LabelEncoder

import numpy as np
import math


def get_word_list(num_folders_start, num_folders_end):
    folder_path = 'morpheme/01'

    # 단어들을 저장할 리스트
    word_list = []
    
    # 파일 이름 얻어오기
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.json') and "F_morpheme" in f]

    # 파일 이름을 번호 순서대로 정렬하기
    file_names.sort(key=lambda x: int(x.split('_')[2][4:]))

    file_names = file_names[:500]
    
    for idx in range(num_folders_start, num_folders_end + 1):
        for filename in file_names:
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # 'data' 키 안의 요소들 순회
                for item in data['data']:
                    for attribute in item['attributes']:
                        word_list.append(attribute['name'])
                    

    # Label Encoder 초기화 및 학습
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(word_list)
    
    return encoded_labels, label_encoder


def get_sequence_files(num_folders_start, num_folders_end):
    base_folder_path = 'keypoints'

    # 전체 시퀀스를 저장할 리스트
    sequence_files = []

    for idx in range(num_folders_start, num_folders_end + 1):

        folder_path = os.path.join(base_folder_path, f'{idx:02d}')
        
        # 각 폴더의 파일 이름을 저장할 리스트
        # folder_files = []
        
        # 파일 이름 얻어오기
        file_names = [f for f in os.listdir(folder_path) if "F" in f]
        
        # 파일 이름을 번호 순서대로 정렬하기
        file_names.sort(key=lambda x: int(x.split('_')[2][4:]))
        
        file_names = file_names[:500]
        
        for filename in file_names:
            file_path = os.path.join(folder_path, filename)
            
            json_names = [f for f in os.listdir(file_path) if "F" in f]       
            json_names.sort(key=lambda x: int(x.split('_')[5]))
            json_paths = []
            
            for i, jsonname in enumerate(json_names):
                if i % 3 == 0:
                    json_path = os.path.join(file_path, jsonname)
                    json_paths.append(json_path)
            
            sequence_files.append(json_paths)
        
    return sequence_files
                

def extract_keypoints(json_data):
    
    def append_coordinates(keypoints_list, array, dimensions, offset=0):
        step = dimensions + 1  # dimensions + 1 because of the confidence score
        for i in range(0, len(keypoints_list), step):
            idx = i // step + offset
            if dimensions == 2:
                array[idx] = [keypoints_list[i], keypoints_list[i + 1], 0]
            elif dimensions == 3:
                array[idx] = [keypoints_list[i], keypoints_list[i + 1], keypoints_list[i + 2]]
                
    
    keypoint_types_2d = ['face_keypoints_2d', 'pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    keypoint_types_3d = ['face_keypoints_3d', 'pose_keypoints_3d', 'hand_left_keypoints_3d', 'hand_right_keypoints_3d']
    
    #키포인트 개수 계산
    num_keypoints_2d = sum(len(json_data['people'][key]) // 3 for key in keypoint_types_2d if key in json_data['people'])
    num_keypoints_3d = sum(len(json_data['people'][key]) // 4 for key in keypoint_types_3d if key in json_data['people'])

    if num_keypoints_3d == 0: 
        #3d가 비어있을 때
        #numpy배열 초기화
        keypoints_2d = np.zeros((num_keypoints_2d, 3))  # (x, y, 0)
        
        offset_2d = 0
        
        for key in keypoint_types_2d:
            if key in json_data['people']:
                append_coordinates(json_data['people'][key], keypoints_2d, dimensions=2, offset=offset_2d)
                offset_2d += len(json_data['people'][key]) // 3
                
        keypoints = keypoints_2d
        
    else:
        #3d가 있을 때
        #키포인트 개수 계산

        #numpy배열 초기화
        keypoints_3d = np.zeros((num_keypoints_3d, 3))  # (x, y, z)

        offset_3d = 0

        for key in keypoint_types_3d:
            if key in json_data['people']:
                append_coordinates(json_data['people'][key], keypoints_3d, dimensions=3, offset=offset_3d)
                offset_3d += len(json_data['people'][key]) // 4

        keypoints = keypoints_3d
                
    return keypoints


# # Collate 함수 정의
# def collate_fn(batch):
#     # batch는 keypoints와 labels의 튜플로 구성된 리스트
#     keypoints, labels = zip(*batch)
    
#     # keypoints는 3D 텐서이므로, 텐서 리스트에서 시퀀스 길이(120)를 추출하여 패딩 처리
#     keypoints_padded = pad_sequence([k.permute(1, 0, 2) for k in keypoints], batch_first=True, padding_value=0)
    
#     # 패딩 후 다시 원래 차원으로 복원
#     keypoints_padded = keypoints_padded.permute(0, 2, 1, 3)
    
#     # 각 시퀀스의 길이를 계산 (여기서는 모두 120이 동일함)
#     lengths = torch.tensor([k.size(1) for k in keypoints])
    
#     # labels를 tensor로 변환
#     labels = torch.tensor(labels)
    
#     return keypoints_padded, labels, lengths

def collate_fn(batch):    
    # batch는 keypoints와 labels의 튜플로 구성된 리스트
    keypoints, labels, idx = zip(*batch)
        
    # 각 keypoints 리스트에서 face, pose, left_hand, right_hand를 각각 분리
    face_keypoints = [k[0] for k in keypoints]
    pose_keypoints = [k[1] for k in keypoints]
    left_hand_keypoints = [k[2] for k in keypoints]
    right_hand_keypoints = [k[3] for k in keypoints]
    
    # keypoints는 3D 텐서이므로, 텐서 The line `리스트에서 시퀀스 길이(120)를 추출
    face_keypoints_padded = pad_sequence([k.permute(1, 0, 2) for k in face_keypoints], batch_first=True, padding_value=0)
    pose_keypoints_padded = pad_sequence([k.permute(1, 0, 2) for k in pose_keypoints], batch_first=True, padding_value=0)
    left_hand_keypoints_padded = pad_sequence([k.permute(1, 0, 2) for k in left_hand_keypoints], batch_first=True, padding_value=0)
    right_hand_keypoints_padded = pad_sequence([k.permute(1, 0, 2) for k in right_hand_keypoints], batch_first=True, padding_value=0)
    
    # 패딩 후 다시 원래 차원으로 복원
    face_keypoints_padded = face_keypoints_padded.permute(0, 2, 1, 3)
    pose_keypoints_padded = pose_keypoints_padded.permute(0, 2, 1, 3)
    left_hand_keypoints_padded = left_hand_keypoints_padded.permute(0, 2, 1, 3)
    right_hand_keypoints_padded = right_hand_keypoints_padded.permute(0, 2, 1, 3)
    
    # 각 시퀀스의 길이를 계산 (여기서는 모두 120이 동일함)
    lengths = torch.tensor([k.size(1) for k in keypoints[0]])
    
    
    # labels를 tensor로 변환
    labels = torch.tensor(labels)
    
    return (face_keypoints_padded,pose_keypoints_padded,left_hand_keypoints_padded,right_hand_keypoints_padded), labels, lengths, idx


class SignLanguageDataset(Dataset):
    def __init__(self, sequence_files, labels):
        self.data = []
        self.labels = labels
        for files in sequence_files:
            sequence = []
            for file in files:
                with open(file, 'r') as f:
                    json_data = json.load(f)
                    keypoints = extract_keypoints(json_data)
                    sequence.append(keypoints)
            self.data.append(sequence)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.data[idx], dtype=torch.float32)        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        sequence = sequence.permute(2, 0, 1)
        
        # 데이터 분리: 얼굴, 왼손, 오른손, 포즈 (각각의 키포인트 개수는 데이터에 따라 설정)
        face_sequence = sequence[:, :, :70]  # 얼굴 keypoints
        pose_sequence = sequence[:, :, 70:95]  # 포즈 keypoints
        left_hand_sequence = sequence[:, :, 95:116]  # 왼손 keypoints
        right_hand_sequence = sequence[:, :, 116:137]  # 오른손 keypoints
        
        mean = torch.mean(face_sequence)
        std = torch.std(face_sequence)
        face_sequence = (face_sequence - mean) / std
        
        mean = torch.mean(pose_sequence)
        std = torch.std(pose_sequence)
        pose_sequence = (pose_sequence - mean) / std
        
        mean = torch.mean(left_hand_sequence)
        std = torch.std(left_hand_sequence)
        left_hand_sequence = (left_hand_sequence - mean) / std
        
        mean = torch.mean(right_hand_sequence)
        std = torch.std(right_hand_sequence)
        right_hand_sequence = (right_hand_sequence - mean) / std
        
        # 패딩 처리: 길이를 70으로 맞춤
        pose_sequence = F.pad(pose_sequence, (0, 70 - pose_sequence.size(2)), "constant", 0)
        left_hand_sequence = F.pad(left_hand_sequence, (0, 70 - left_hand_sequence.size(2)), "constant", 0)
        right_hand_sequence = F.pad(right_hand_sequence, (0, 70 - right_hand_sequence.size(2)), "constant", 0)

        # 마스크 생성: 패딩된 부분은 True, 나머지는 False
        pose_mask = pose_sequence.sum(dim=1) == 0  # 패딩된 부분은 모든 값이 0이므로 합이 0인 부분이 패딩된 부분
        left_hand_mask = left_hand_sequence.sum(dim=1) == 0
        right_hand_mask = right_hand_sequence.sum(dim=1) == 0
    
        return (face_sequence, pose_sequence, left_hand_sequence, right_hand_sequence), label, idx

class MultiEncoderTransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes):
        super(MultiEncoderTransformerModel, self).__init__()
        
        # Face Encoder
        self.face_input_fc = nn.Linear(input_dim, model_dim)
        self.face_positional_encoding = PositionalEncoding(dim_model=model_dim, dropout_p=0.1, max_len=500)
        face_encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.face_transformer_encoder = nn.TransformerEncoder(face_encoder_layers, num_layers)
        
        # Pose Encoder
        self.pose_input_fc = nn.Linear(input_dim, model_dim)
        self.pose_positional_encoding = PositionalEncoding(dim_model=model_dim, dropout_p=0.1, max_len=500)
        pose_encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.pose_transformer_encoder = nn.TransformerEncoder(pose_encoder_layers, num_layers)
        
        # Left Hand Encoder
        self.left_hand_input_fc = nn.Linear(input_dim, model_dim)
        self.left_hand_positional_encoding = PositionalEncoding(dim_model=model_dim, dropout_p=0.1, max_len=500)
        left_hand_encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.left_hand_transformer_encoder = nn.TransformerEncoder(left_hand_encoder_layers, num_layers)
        
        # Right Hand Encoder
        self.right_hand_input_fc = nn.Linear(input_dim, model_dim)
        self.right_hand_positional_encoding = PositionalEncoding(dim_model=model_dim, dropout_p=0.1, max_len=500)
        right_hand_encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.right_hand_transformer_encoder = nn.TransformerEncoder(right_hand_encoder_layers, num_layers)
        
        # 최종 결합 후 클래스 분류
        self.fc = nn.Linear(model_dim * 4, num_classes)  # 네 인코더 출력 결합

    def forward(self, face_input, pose_input, left_hand_input, right_hand_input, face_src_key_padding_mask=None, pose_src_key_padding_mask=None, left_src_key_padding_mask=None, right_src_key_padding_mask=None):
        # Face Encoder
        face_x = self.face_input_fc(face_input)
        face_x = self.face_positional_encoding(face_x)
        face_x = self.face_transformer_encoder(face_x, src_key_padding_mask=face_src_key_padding_mask)
        face_x = face_x.mean(dim=1)
        
        # Pose Encoder
        pose_x = self.pose_input_fc(pose_input)
        pose_x = self.pose_positional_encoding(pose_x)
        pose_x = self.pose_transformer_encoder(pose_x, src_key_padding_mask=pose_src_key_padding_mask)
        pose_x = pose_x.mean(dim=1)
        
        # Left Hand Encoder
        left_hand_x = self.left_hand_input_fc(left_hand_input)
        left_hand_x = self.left_hand_positional_encoding(left_hand_x)
        left_hand_x = self.left_hand_transformer_encoder(left_hand_x, src_key_padding_mask=left_src_key_padding_mask)
        left_hand_x = left_hand_x.mean(dim=1)
        
        # Right Hand Encoder
        right_hand_x = self.right_hand_input_fc(right_hand_input)
        right_hand_x = self.right_hand_positional_encoding(right_hand_x)
        right_hand_x = self.right_hand_transformer_encoder(right_hand_x, src_key_padding_mask=right_src_key_padding_mask)
        right_hand_x = right_hand_x.mean(dim=1)
        
        # 네 인코더의 출력 결합
        combined_x = torch.cat((face_x, left_hand_x, right_hand_x, pose_x), dim=-1)
        
        # Fully Connected Layer
        output = self.fc(combined_x)
        
        return output


# class TransformerModel(nn.Module):
#     def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes):
#         super(TransformerModel, self).__init__()
#         self.input_fc = nn.Linear(input_dim, model_dim)
        
#         # Positional Encoding 추가
#         self.positional_encoding = PositionalEncoding(dim_model=model_dim, dropout_p=0.1, max_len=500)
        
#         encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
#         self.fc = nn.Linear(model_dim, num_classes)
#         self.relu = nn.ReLU()  # 추가된 ReLU 활성화 함수

    
#     def forward(self, x, src_key_padding_mask):
#         x = self.input_fc(x)
#         # x = self.relu(x)  # 활성화 함수 적용
#         # Positional Encoding 적용
#         x = self.positional_encoding(x)
        
#         x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # transformer 대신 transformer_encoder 사용
#         x = x.mean(dim=1)  # 시퀀스 차원 축소
#         x = self.fc(x)
#         return x
    


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len=5000):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p)
        
        # 최대 길이에 대한 Positional Encoding 생성
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, ...
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Positional Encoding을 모델의 버퍼로 등록
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
    
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        seq_len = token_embedding.size(1)
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        return self.dropout(token_embedding + pos_encoding)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True