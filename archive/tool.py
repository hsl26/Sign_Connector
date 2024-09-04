import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import os
import json
from sklearn.preprocessing import LabelEncoder

import numpy as np
import math


def get_word_list(num_folders_start=1, num_folders_end=7):
    folder_path = 'morpheme/01'

    # 단어들을 저장할 리스트
    word_list = []
    
    # 파일 이름 얻어오기
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.json') and "F_morpheme" in f]

    # 파일 이름을 번호 순서대로 정렬하기
    file_names.sort(key=lambda x: int(x.split('_')[2][4:]))

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
    
    return encoded_labels


def get_sequence_files(num_folders_start=1, num_folders_end=7):
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
        
        for filename in file_names:
            file_path = os.path.join(folder_path, filename)
            
            json_names = [f for f in os.listdir(file_path) if "F" in f]       
            json_names.sort(key=lambda x: int(x.split('_')[5]))
            
            for i, jsonname in enumerate(json_names):
                json_path = os.path.join(file_path, jsonname)
                json_names[i] = json_path
            
            sequence_files.append(json_names)
        
    return sequence_files
                

def extract_keypoints(json_data):
    keypoint_types_2d = ['face_keypoints_2d', 'pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    keypoint_types_3d = ['face_keypoints_3d', 'hand_left_keypoints_3d', 'hand_right_keypoints_3d']

    # 2D와 3D 키포인트의 총 개수를 계산
    num_keypoints_2d = sum(len(json_data['people'][key]) // 3 for key in keypoint_types_2d if key in json_data['people'])
    num_keypoints_3d = sum(len(json_data['people'][key]) // 4 for key in keypoint_types_3d if key in json_data['people'])

    # Numpy 배열 초기화
    keypoints_2d = np.zeros((num_keypoints_2d, 3))  # (x, y, 0)
    keypoints_3d = np.zeros((num_keypoints_3d, 3))  # (x, y, z)

    def append_coordinates(keypoints_list, array, dimensions, offset=0):
        step = dimensions + 1  # dimensions + 1 because of the confidence score
        for i in range(0, len(keypoints_list), step):
            idx = i // step + offset
            if dimensions == 2:
                array[idx] = [keypoints_list[i], keypoints_list[i + 1], 0]
            elif dimensions == 3:
                array[idx] = [keypoints_list[i], keypoints_list[i + 1], keypoints_list[i + 2]]

    offset_2d = 0
    offset_3d = 0

    for key in keypoint_types_2d:
        if key in json_data['people']:
            append_coordinates(json_data['people'][key], keypoints_2d, dimensions=2, offset=offset_2d)
            offset_2d += len(json_data['people'][key]) // 3

    for key in keypoint_types_3d:
        if key in json_data['people']:
            append_coordinates(json_data['people'][key], keypoints_3d, dimensions=3, offset=offset_3d)
            offset_3d += len(json_data['people'][key]) // 4

    # 필요에 따라 keypoints_2d와 keypoints_3d를 하나의 배열로 합칠 수 있음
    keypoints = np.vstack((keypoints_2d, keypoints_3d))
    
    return keypoints

# Collate 함수 정의
def collate_fn(batch):
    # batch는 keypoints와 labels의 튜플로 구성된 리스트
    keypoints, labels = zip(*batch)
    
    # keypoints는 3D 텐서이므로, 텐서 리스트에서 시퀀스 길이(120)를 추출하여 패딩 처리
    keypoints_padded = pad_sequence([k.permute(1, 0, 2) for k in keypoints], batch_first=True, padding_value=0)
    
    # 패딩 후 다시 원래 차원으로 복원
    keypoints_padded = keypoints_padded.permute(0, 2, 1, 3)
    
    # 각 시퀀스의 길이를 계산 (여기서는 모두 120이 동일함)
    lengths = torch.tensor([k.size(1) for k in keypoints])
    
    # labels를 tensor로 변환
    labels = torch.tensor(labels)
    
    return keypoints_padded, labels, lengths


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
        return sequence, label


class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        
        # Positional Encoding 추가
        self.positional_encoding = PositionalEncoding(dim_model=model_dim, dropout_p=0.1, max_len=500)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(model_dim, num_classes)
        self.relu = nn.ReLU()  # 추가된 ReLU 활성화 함수

    
    def forward(self, x, src_key_padding_mask):
        x = self.input_fc(x)
        x = self.relu(x)  # 활성화 함수 적용
        # Positional Encoding 적용
        x = self.positional_encoding(x)
        
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # transformer 대신 transformer_encoder 사용
        x = x.mean(dim=1)  # 시퀀스 차원 축소
        x = self.fc(x)
        return x
    


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