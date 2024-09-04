import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import LabelEncoder

import os
import json

import numpy as np


def get_word_list(start, end):
    folder_path = 'morpheme/01'

    # 단어들을 저장할 리스트
    word_list = []

    # 파일 이름 얻어오기
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.json') and "F_morpheme" in f]

    # 파일 이름을 번호 순서대로 정렬하기
    file_names.sort(key=lambda x: int(x.split('_')[2][4:]))

    for filename in file_names:
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # 'data' 키 안의 요소들 순회
            for item in data['data']:
                for attribute in item['attributes']:
                    word_list.append(attribute['name'])

    # 결과 출력 
    #print(len(word_list))
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(word_list)
    # label_mapping 딕셔너리 생성
    folder_to_label = {}
    for i, word in enumerate(word_list):
        for j in range(start, end+1):  # 01부터 07까지
            folder_name = f"NIA_SL_WORD{str(i+1).zfill(4)}_REAL{str(j).zfill(2)}_F"
            folder_to_label[folder_name] = encoded_labels[i]
    
    return folder_to_label, label_encoder, word_list

    
# 리얼리얼 임시
class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, folder_to_label, start, end, label_encoder):
        self.data_dir = data_dir
        self.folder_to_label = folder_to_label
        self.load_num_start = start
        self.load_num_end = end
        self.label_encoder = label_encoder
        self.data, self.labels = self.load_data()

    def load_data(self):
        file_list = []
        labels = []
        for subdir in range(self.load_num_start, self.load_num_end + 1):  # 01~07 디렉토리
            subdir_path = os.path.join(self.data_dir, f'{subdir:02d}')
            for folder_name in os.listdir(subdir_path):
                if folder_name.endswith("F") and folder_name in self.folder_to_label:  # "F"로 끝나는 폴더만 처리
                    label = self.folder_to_label[folder_name]
                    label = int(label)
                    label_name = self.label_encoder.inverse_transform([label])
                    folder_path = os.path.join(subdir_path, folder_name)
                    if os.path.isdir(folder_path):
                        json_files = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.json')]
                        file_list.append(json_files)
                        labels.append(label)
        return file_list, labels
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        all_keypoints = []
        
        json_file_list = self.data[index]
        label = self.labels[index]
        for file_path in json_file_list:
            with open(file_path, 'r') as f:
                data = json.load(f)
            frame_data = data['people']
            if frame_data:
                keypoints_2d = np.array(frame_data['face_keypoints_2d'] +
                                        frame_data['pose_keypoints_2d'] +
                                        frame_data['hand_left_keypoints_2d'] +
                                        frame_data['hand_right_keypoints_2d'])
                keypoints_2d = keypoints_2d.reshape(-1, 3)[:, :2].flatten()  # (num_keypoints * 2,)
                
                keypoints_3d = np.array(frame_data['face_keypoints_3d'] +
                                        frame_data['pose_keypoints_3d'] +
                                        frame_data['hand_left_keypoints_3d'] +
                                        frame_data['hand_right_keypoints_3d'])
                keypoints_3d = keypoints_3d.reshape(-1, 4)[:, :3].flatten() 
                
                keypoints = np.concatenate((keypoints_2d, keypoints_3d))
                all_keypoints.append(keypoints)
        
        all_keypoints = np.array(all_keypoints)  # (num_frames, num_keypoints * (2 + 3))
        
        return torch.tensor(all_keypoints, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
# Collate 함수 정의
def collate_fn(batch):
    keypoints, labels = zip(*batch)
    keypoints = [torch.tensor(k) for k in keypoints]
    labels = torch.tensor(labels)
    keypoints_padded = pad_sequence(keypoints, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(k) for k in keypoints])
    return keypoints_padded, labels, lengths

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(model_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()  # 추가된 ReLU 활성화 함수
    
    def forward(self, x, src_key_padding_mask=None):
        x = self.input_fc(x)
        x = self.relu(x)  # 활성화 함수 적용
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.mean(dim=1)  # 시퀀스 차원 축소
        x = self.fc(x)
        x = self.log_softmax(x)
        return x

def create_padding_mask(sequences, pad_token=0):
    return (sequences == pad_token)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
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