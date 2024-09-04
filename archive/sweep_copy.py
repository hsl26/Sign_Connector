import tool_copy as tl

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# device = torch.device('cpu')  # CPU로 강제 설정

# sweep 조건(?)들
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size' : {
            'values' : [16, 32, 64]
        },
        'model_dim': {
            'values': [128, 256, 512]
        },
        'num_heads': {
            'values': [4, 8, 16]
        },
        'num_layers': {
            'values': [2, 3, 4]
        },
        'learning_rate': {
            'max': 0.01,
            'min': 0.0001
        },
        'num_epochs': {
            'values': [500]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project='sign-language-transformer')

    
def train():
    wandb.init()
    
    root_dir = "keypoints"

    train_folder_to_label, train_label_encoder, train_word_list = tl.get_word_list(1, 4)
    train_dataset = tl.SignLanguageDataset(root_dir, train_folder_to_label, 1, 4, train_label_encoder)
    train_loader = DataLoader(train_dataset, wandb.config.batch_size, collate_fn=tl.collate_fn)  

    val_folder_to_label, val_label_encoder, val_word_list = tl.get_word_list(8, 8)
    val_dataset = tl.SignLanguageDataset(root_dir, val_folder_to_label, 8, 8, val_label_encoder)
    val_loader = DataLoader(val_dataset, wandb.config.batch_size, collate_fn=tl.collate_fn) 

    # 모델 하이퍼파라미터 설정
    num_keypoints = 138  # face, pose, hand_left, hand_right keypoints 필요없어 
    input_dim = 685 # 각 키포인트의 2D 좌표(2)와 3D 좌표(3)를 사용
    model_dim = 128  # 모델 차원
    num_heads = 8  # 멀티헤드 어텐션의 헤드 수
    num_layers = 4  # Transformer 레이어 수
    num_classes = 2771  # 출력 클래스 수
    learning_rate = wandb.config.learning_rate
    num_epochs = wandb.config.num_epochs

    model = tl.TransformerModel(input_dim, model_dim, num_heads, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    early_stopping = tl.EarlyStopping(patience=50, min_delta=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for sequences, labels, lengths in train_loader:
            
            mean = torch.mean(sequences)
            std = torch.std(sequences)
            sequences = (sequences - mean) / std
            
            sequences = sequences.to(device)
            labels = labels.to(device)
            src_key_padding_mask = (sequences.sum(dim=-1) == 0).to(device)
            
            # 입력 텐서의 차원을 출력하여 확인
            # print(f"Sequences shape: {sequences.shape}")
            # print(f"Labels shape: {labels.shape}")

            # Forward, Backward, Optimize
            optimizer.zero_grad()
            outputs = model(sequences, src_key_padding_mask=src_key_padding_mask)
        
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels, lengths in val_loader:
                
                mean = torch.mean(sequences)
                std = torch.std(sequences)
                sequences = (sequences - mean) / std
                
                sequences = sequences.to(device)
                labels = labels.to(device)
                # 패딩 마스크 생성
                src_key_padding_mask = tl.create_padding_mask(sequences[:,:,0])
                src_key_padding_mask = src_key_padding_mask.to(device)

                outputs = model(sequences, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 에폭당 loss 값을 기록합니다.
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
        wandb.log({"epoch": epoch+1, "loss": loss.item(), "val_loss": val_loss})
        
        # Early stopping 체크
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    print('Training finished.')


wandb.agent(sweep_id, train)