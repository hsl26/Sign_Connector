import streamlit as st
import os
import json
import re
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, random_split

import torch.optim as optim

import os
import json
from sklearn.preprocessing import LabelEncoder

import numpy as np
import math

import tool as tl

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

model_path = 'model/final_model_1000_red.pth'
input_dim, model_dim, num_heads, num_layers, num_classes, learning_rate = 70*3, 512, 8, 2, 501, 0.00001
checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
model = tl.MultiEncoderTransformerModel(input_dim, model_dim, num_heads, num_layers, num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval() 

# GPU 사용 여부 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit 인터페이스 설정
st.set_page_config(page_title="Sign Connector\n(수어 번역 도우미)", layout="centered")

# 맞춤 CSS 적용
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .title {
        font-family: 'Arial', sans-serif;
        color: #343a40;
        text-align: center;
        font-weight: bold;
        font-size: 48px;
    }
    .subtitle {
        font-family: 'Arial', sans-serif;
        color: #6c757d;
        text-align: center;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .upload-box {
        margin-top: 50px;
        border: 2px dashed #6c757d;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        background-color: #e9ecef;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #adb5bd;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# JSON 파일 업로드 섹션
uploaded_files = st.file_uploader("동영상을 업로드하세요.", type="json", accept_multiple_files=True)
# 저장할 디렉토리 지정
save_directory = "keypoints/11/NIA_SL_WORD0001_REAL11_F" 

i=1
# 파일 저장
if uploaded_files:
    for uploaded_file in uploaded_files:
        # 저장할 파일 경로 설정
        save_path = os.path.join(save_directory, uploaded_file.name)
        
        # 파일을 저장
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        i += 1
    st.success(f"{i}개의 파일이 저장되었습니다.")
    
    test_encoded_labels, encoder = tl.get_word_list(11, 11)
    test_sequence_files = tl.get_sequence_files(11,11)

    test_dataset = tl.SignLanguageDataset(test_sequence_files, test_encoded_labels)  # 실제 테스트 데이터셋 경로와 라벨 사용
    test_loader = tl.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=tl.collate_fn)

    def create_padding_mask(sequences, pad_token=0):
            return (sequences == pad_token)
                
    all_labels = []
    prediction = 1

    with torch.no_grad():
        for (face_sequence, pose_sequence, left_hand_sequence, right_hand_sequence), labels, lengths, idx in test_loader:
            
            face_sequence, pose_sequence, left_hand_sequence, right_hand_sequence = face_sequence.to(device), pose_sequence.to(device), left_hand_sequence.to(device), right_hand_sequence.to(device)
            labels = labels.to(device)

            # 입력 텐서 변환: [batch_size, 3, seq_len, num_joints] -> [batch_size, seq_len, 3 * num_joints]
            batch_size, coord, seq_len, num_joints = face_sequence.size()
            face_sequence = face_sequence.permute(0, 2, 3, 1).contiguous()  # [batch_size, seq_len, num_joints, coord]
            face_sequence = face_sequence.view(batch_size, seq_len, -1)  # [batch_size, seq_len, num_joints * coord]
            
            batch_size, coord, seq_len, num_joints = pose_sequence.size()
            pose_sequence = pose_sequence.permute(0, 2, 3, 1).contiguous()  # [batch_size, seq_len, num_joints, coord]
            pose_sequence = pose_sequence.view(batch_size, seq_len, -1)  # [batch_size, seq_len, num_joints * coord]
            
            batch_size, coord, seq_len, num_joints = left_hand_sequence.size()
            left_hand_sequence = left_hand_sequence.permute(0, 2, 3, 1).contiguous()  # [batch_size, seq_len, num_joints, coord]
            left_hand_sequence = left_hand_sequence.view(batch_size, seq_len, -1)  # [batch_size, seq_len, num_joints * coord]
            
            batch_size, coord, seq_len, num_joints = right_hand_sequence.size()
            right_hand_sequence = right_hand_sequence.permute(0, 2, 3, 1).contiguous()  # [batch_size, seq_len, num_joints, coord]
            right_hand_sequence = right_hand_sequence.view(batch_size, seq_len, -1)  # [batch_size, seq_len, num_joints * coord]
            
            # 패딩 마스크 생성
            face_src_key_padding_mask = create_padding_mask(face_sequence[:,:,0])
            face_src_key_padding_mask = face_src_key_padding_mask.to(device)
            pose_src_key_padding_mask = create_padding_mask(pose_sequence[:,:,0])
            pose_src_key_padding_mask = pose_src_key_padding_mask.to(device)
            left_src_key_padding_mask = create_padding_mask(left_hand_sequence[:,:,0])
            left_src_key_padding_mask = left_src_key_padding_mask.to(device)
            right_src_key_padding_mask = create_padding_mask(right_hand_sequence[:,:,0])
            right_src_key_padding_mask = right_src_key_padding_mask.to(device)

            outputs = model(face_sequence, pose_sequence, left_hand_sequence, right_hand_sequence, face_src_key_padding_mask, pose_src_key_padding_mask, left_src_key_padding_mask, right_src_key_padding_mask)
            
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_classes = torch.max(probabilities, 1)
            
            prediction = predicted_classes.cpu().numpy()
            label = labels.cpu().numpy()
            
            # 인코딩된 라벨을 디코딩하여 원래의 라벨로 변환
            original_prediction = encoder.inverse_transform(prediction)
    
            
    print(f"디버깅용 extract_keypoints가 리턴한 시퀀스(프레임(키포인트만 담겨있음)) 갯수 : {len(uploaded_files)}, {prediction}")
    
    st.markdown("수어 번역 결과")
    
    container = st.container(border=True)
    container.write(f"{original_prediction[0]}")


else:
    st.markdown('<div style="text-align: center;">JSON 파일들을 업로드하세요.</div>', unsafe_allow_html=True)
    

# 푸터
st.markdown('<div class="footer">© 2024 Sign Connector. All rights reserved.</div>', unsafe_allow_html=True)