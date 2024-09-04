import tool as tl

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
    # 'name': 'hs_21000_6',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size' : {
            'values' : [32, 64]
        },
        'model_dim': {
            'values': [512]
        },
        'num_heads': {
            'values': [8]
        },
        'num_layers': {
            'values': [2, 3, 4]
        },
        'learning_rate': {
            # 'max': 0.01,
            # 'min': 0.0001
            'values': [0.00001]
        },
        'num_epochs': {
            'values': [1000]
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project='sign-language-transformer')

    
def train():
    wandb.init()
    
    input_dim = 70 * 3 # 각 키포인트의 2D 좌표(2)와 3D 좌표(3)를 사용
    model_dim = wandb.config.model_dim
    num_heads = wandb.config.num_heads
    num_layers = wandb.config.num_layers
    num_classes = 501  # 출력 클래스 수
    learning_rate = wandb.config.learning_rate
    num_epochs = wandb.config.num_epochs
    batch_size = wandb.config.batch_size
    
    train_encoded_labels, enc = tl.get_word_list(1, 8)
    train_sequence_files = tl.get_sequence_files(1, 8)

    val_encoded_labels, enc = tl.get_word_list(9, 9)
    val_sequence_files = tl.get_sequence_files(9, 9)

    train_dataset = tl.SignLanguageDataset(train_sequence_files, train_encoded_labels)
    val_dataset = tl.SignLanguageDataset(val_sequence_files, val_encoded_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=tl.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  collate_fn=tl.collate_fn)

    model = tl.MultiEncoderTransformerModel(input_dim, model_dim, num_heads, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    early_stopping = tl.EarlyStopping(patience=400, min_delta=0.001)

    fin_epoch = 0
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for (face_sequence, pose_sequence, left_hand_sequence, right_hand_sequence), labels, lengths, idx in train_loader:
            
            face_sequence = face_sequence.to(device)
            pose_sequence = pose_sequence.to(device)
            left_hand_sequence = left_hand_sequence.to(device)
            right_hand_sequence = right_hand_sequence.to(device)
            labels = labels.to(device)
            
            # 마스킹 생성
            face_src_key_padding_mask = (face_sequence.sum(dim=-1) == 0)
            face_src_key_padding_mask = face_src_key_padding_mask.any(dim=1).to(device)
            pose_src_key_padding_mask = (pose_sequence.sum(dim=-1) == 0)
            pose_src_key_padding_mask = pose_src_key_padding_mask.any(dim=1).to(device)
            left_src_key_padding_mask = (left_hand_sequence.sum(dim=-1) == 0)
            left_src_key_padding_mask = left_src_key_padding_mask.any(dim=1).to(device)
            right_src_key_padding_mask = (right_hand_sequence.sum(dim=-1) == 0)
            right_src_key_padding_mask = right_src_key_padding_mask.any(dim=1).to(device)

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
            

            # Forward pass
            outputs = model(face_sequence, pose_sequence, left_hand_sequence, right_hand_sequence, face_src_key_padding_mask, pose_src_key_padding_mask, left_src_key_padding_mask, right_src_key_padding_mask)
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
            for (face_sequence, pose_sequence, left_hand_sequence, right_hand_sequence), labels, lengths, idx in val_loader:
                
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
                # outputs = model(sequences, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 에폭당 loss 값을 기록합니다.
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
        wandb.log({"epoch": epoch+1, "loss": loss.item(), "val_loss": val_loss})
        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        # wandb.log({"epoch": epoch+1, "loss": loss.item()})
        
        fin_epoch = epoch+1
        
        # Early stopping 체크
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save the final model at the end of training
    final_save_path = f"model/final_model.pth_{fin_epoch}"
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, final_save_path)
    print(f"Final model saved at {final_save_path}")
    
    print('Training finished.')


wandb.agent(sweep_id, train)