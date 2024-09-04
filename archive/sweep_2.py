import tool as tl

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

train_encoded_labels = tl.get_word_list(1, 7)
train_sequence_files = tl.get_sequence_files(1, 7)

val_encoded_labels = tl.get_word_list(8, 9)
val_sequence_files = tl.get_sequence_files(8, 9)

train_dataset = tl.SignLanguageDataset(train_sequence_files, train_encoded_labels)
val_dataset = tl.SignLanguageDataset(val_sequence_files, val_encoded_labels)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# device = torch.device('cpu')  # CPU로 강제 설정

# sweep 조건(?)들
sweep_config = {
    'method': 'bayes',
    'name': 'hs_21000_ver2_2',
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
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project='sign-language-transformer')

    
def train():
    wandb.init()
    
    input_dim = 249 * 3 # 각 키포인트의 2D 좌표(2)와 3D 좌표(3)를 사용
    model_dim = wandb.config.model_dim
    num_heads = wandb.config.num_heads
    num_layers = wandb.config.num_layers
    num_classes = 2771  # 출력 클래스 수
    learning_rate = wandb.config.learning_rate
    num_epochs = wandb.config.num_epochs
    batch_size = wandb.config.batch_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=tl.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  collate_fn=tl.collate_fn)

    model = tl.TransformerModel(input_dim, model_dim, num_heads, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
            
            # 마스킹 생성
            src_key_padding_mask = (sequences.sum(dim=-1) == 0)
            src_key_padding_mask = src_key_padding_mask.any(dim=1).to(device)

            # 입력 텐서 변환: [batch_size, 3, seq_len, num_joints] -> [batch_size, seq_len, 3 * num_joints]
            batch_size, coord, seq_len, num_joints = sequences.size()
            sequences = sequences.permute(0, 2, 3, 1).contiguous()  # [batch_size, seq_len, num_joints, coord]
            sequences = sequences.view(batch_size, seq_len, -1)  # [batch_size, seq_len, num_joints * coord]
            
            # Forward pass
            outputs = model(sequences, src_key_padding_mask=src_key_padding_mask)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        # Validation loop
        model.eval()
        val_loss = 0
        
        def create_padding_mask(sequences, pad_token=0):
            return (sequences == pad_token)
        
        with torch.no_grad():
            for sequences, labels, lengths in val_loader:
                
                mean = torch.mean(sequences)
                std = torch.std(sequences)
                sequences = (sequences - mean) / std
                
                sequences = sequences.to(device)
                labels = labels.to(device)

                batch_size, coord, seq_len, num_joints = sequences.size()
                sequences = sequences.permute(0, 2, 3, 1).contiguous()
                sequences = sequences.view(batch_size, seq_len, -1)
                
                # 패딩 마스크 생성
                src_key_padding_mask = create_padding_mask(sequences[:,:,0])
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