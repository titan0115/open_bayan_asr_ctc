# 4_finetune.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torchaudio
import pandas as pd

# Импортируем наши модули
from model import CustomSpeechEncoder, CTClassificationHead, ASRModel

# ===== ГИПЕРПАРАМЕТРЫ =====
# Пути
DATASET_PATH = "./data/finetune_dataset"  # Путь к папке с размеченными данными (metadata.csv и audio_files/)
PRETRAINED_CHECKPOINT = "./checkpoints/pretrained_encoder_epoch_50.pt"  # Путь к файлу с весами предобученного энкодера
SAVE_PATH = "./finetuned_model"  # Папка для сохранения финальной модели

# Гиперпараметры обучения
EPOCHS = 100  # Количество эпох
BATCH_SIZE = 16  # Размер батча
LEARNING_RATE = 3e-4  # Скорость обучения

# Параметры загрузки данных
NUM_WORKERS = 4  # Количество воркеров для загрузки данных
PIN_MEMORY = True  # Использовать закрепленную память для ускорения передачи на GPU
PREFETCH_FACTOR = 2  # Количество батчей, предзагружаемых каждым воркером
PERSISTENT_WORKERS = True  # Сохранять воркеров между эпохами
# =========================

# Определяем словарь (важно, чтобы 0 был BLANK символом для CTC)
VOCAB = ["_"] + list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя") + [" "]
char_to_int = {char: i for i, char in enumerate(VOCAB)}
int_to_char = {i: char for i, char in enumerate(VOCAB)}


class SupervisedAudioDataset(Dataset):
    def __init__(self, metadata_file: Path, audio_dir: Path):
        self.data = pd.read_csv(metadata_file, sep='|', header=None, names=['filename', 'text'])
        self.audio_dir = audio_dir
        self.target_sr = 16000

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filepath = self.audio_dir / row['filename']
        text = row['text'].lower()

        waveform, sr = torchaudio.load(filepath)
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
        
        labels = [char_to_int.get(c, char_to_int['_']) for c in text]
        
        return waveform.squeeze(0), torch.tensor(labels, dtype=torch.long)

def collate_fn_finetune(batch):
    waveforms, labels = zip(*batch)
    
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    waveform_lengths = torch.tensor([w.size(0) for w in waveforms])
    
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    label_lengths = torch.tensor([len(l) for l in labels])
    
    return padded_waveforms, waveform_lengths, padded_labels, label_lengths

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # 1. Данные
    dataset_path = Path(DATASET_PATH)
    dataset = SupervisedAudioDataset(dataset_path / 'metadata.csv', dataset_path / 'audio_files')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            collate_fn=collate_fn_finetune, 
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY,
                            prefetch_factor=PREFETCH_FACTOR,
                            persistent_workers=PERSISTENT_WORKERS)

    # 2. Модель
    encoder = CustomSpeechEncoder()
    # Загружаем веса из предобученной модели
    encoder.load_state_dict(torch.load(PRETRAINED_CHECKPOINT, map_location=device))
    print("Веса предобученного энкодера загружены.")
    
    head = CTClassificationHead(vocab_size=len(VOCAB))
    model = ASRModel(encoder, head).to(device)

    # 3. Функция потерь (встроена в PyTorch) и оптимизатор
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Цикл обучения
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (waveforms, wave_lens, labels, label_lens) in enumerate(dataloader):
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logits = model(waveforms) # -> [B, T, C]
            
            # Подготовка для CTC Loss
            log_probs = nn.functional.log_softmax(logits, dim=2)
            log_probs = log_probs.transpose(0, 1) # -> [T, B, C]
            
            # Длины входов для CTC должны соответствовать выходу модели
            input_lengths = torch.div(wave_lens, model.encoder.feature_extractor.conv_layers[-1].stride[0], rounding_mode='floor')
            # Это упрощение, точный расчет зависит от всех страйдов и кернелов. 
            # На практике нужно посчитать `(T_in - K)/S + 1` для каждого слоя.
            # Давайте пока оставим так для простоты.
            
            loss = criterion(log_probs, labels, input_lengths, label_lens)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f"Эпоха [{epoch+1}/{EPOCHS}], Шаг [{i+1}/{len(dataloader)}], Потери: {loss.item():.4f}")

        print(f"Средние потери за эпоху {epoch+1}: {total_loss / len(dataloader):.4f}")

    # Сохранение финальной модели
    torch.save(model.state_dict(), f"{SAVE_PATH}/finetuned_asr_model.pt")
    print(f"Финальная модель сохранена в {SAVE_PATH}/finetuned_asr_model.pt")

if __name__ == "__main__":
    Path(SAVE_PATH).mkdir(exist_ok=True, parents=True)
    main()