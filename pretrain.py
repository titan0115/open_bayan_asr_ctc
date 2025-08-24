
# 3_pretrain.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# <--- ИСПРАВЛЕНИЕ: Добавлен недостающий импорт
import torch.amp as amp
from pathlib import Path
import torchaudio
import random
import librosa
import numpy as np

# Импортируем наши модули
from model import CustomSpeechEncoder
from loss import ContrastiveLoss

# ===== ГИПЕРПАРАМЕТРЫ =====
DATASET_PATH = "./data"
SAVE_PATH = "./checkpoints"
D_MODEL = 768
N_HEAD = 12
N_LAYERS = 12
EPOCHS = 50
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
TEMPERATURE = 0.1
USE_AMP = True
USE_CHECKPOINTING = True
SEED = 42

# Параметры загрузки данных
NUM_WORKERS = 4  # Количество воркеров для загрузки данных
PIN_MEMORY = True  # Использовать закрепленную память для ускорения передачи на GPU
PREFETCH_FACTOR = 2  # Количество батчей, предзагружаемых каждым воркером
PERSISTENT_WORKERS = True  # Сохранять воркеров между эпохами
# =========================

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3'}

class UnsupervisedAudioDataset(Dataset):
    def __init__(self, root_dir: Path, segment_duration_sec=5):
        self.files = [p for p in root_dir.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS]
        self.target_sr = 16000
        self.segment_len = self.target_sr * segment_duration_sec

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        try:
            try:
                waveform, sr = torchaudio.load(filepath)
                if waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != self.target_sr:
                    waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
            except Exception:
                waveform, sr = librosa.load(str(filepath), sr=self.target_sr, mono=True)
                waveform = torch.from_numpy(waveform).unsqueeze(0)
            
            if waveform.size(1) > self.segment_len:
                start = random.randint(0, waveform.size(1) - self.segment_len)
                waveform = waveform[:, start:start+self.segment_len]
            
            return waveform.squeeze(0)
        except Exception as e:
            print(f"Ошибка при загрузке файла {filepath}: {e}")
            return torch.zeros(self.segment_len)

def collate_fn_pretrain(batch):
    waveforms = [item for item in batch if torch.sum(torch.abs(item)) > 0]
    if not waveforms:
        return None
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    return padded_waveforms

def apply_mask(cnn_features, mask_prob=0.15, mask_length=10):
    batch_size, dim, seq_len = cnn_features.shape
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=cnn_features.device)
    num_mask_spans = int(mask_prob * seq_len / mask_length)
    
    for i in range(batch_size):
        for _ in range(num_mask_spans):
            start = random.randint(0, max(0, seq_len - mask_length))
            mask[i, start:start+mask_length] = True
            
    return mask

def main():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Используется устройство: {device}")

    dataset = UnsupervisedAudioDataset(Path(DATASET_PATH))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            collate_fn=collate_fn_pretrain, 
                            num_workers=NUM_WORKERS, 
                            pin_memory=PIN_MEMORY,
                            prefetch_factor=PREFETCH_FACTOR,
                            persistent_workers=PERSISTENT_WORKERS)

    model = CustomSpeechEncoder(d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS).to(device)

    criterion = ContrastiveLoss(temperature=TEMPERATURE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    scaler = amp.GradScaler(enabled=(USE_AMP and device.type == 'cuda'))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, waveforms in enumerate(dataloader):
            if waveforms is None:
                continue
                
            waveforms = waveforms.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast(device_type=device_str, dtype=torch.float16, enabled=(USE_AMP and device.type == 'cuda')):
                # Единственный проход через модель, который возвращает все необходимое
                transformer_output, cnn_features_true = model(waveforms, use_checkpointing=USE_CHECKPOINTING)
                
                mask = apply_mask(cnn_features_true)

                if not mask.any():
                    continue
                
                transformer_masked_outputs = transformer_output[mask]
                cnn_masked_targets = cnn_features_true.transpose(1, 2)[mask]

                projected_targets = model.input_projection(cnn_masked_targets)

                loss = criterion(transformer_masked_outputs.unsqueeze(0), projected_targets.unsqueeze(0))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f"Эпоха [{epoch+1}/{EPOCHS}], Шаг [{i+1}/{len(dataloader)}], Потери: {loss.item():.4f}")

        print(f"Средние потери за эпоху {epoch+1}: {total_loss / len(dataloader):.4f}")

        Path(SAVE_PATH).mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), f"{SAVE_PATH}/pretrained_encoder_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()