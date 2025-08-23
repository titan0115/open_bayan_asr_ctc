# 3_pretrain.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torchaudio
import argparse
import random

# Импортируем наши модули
from model import CustomSpeechEncoder
from loss import ContrastiveLoss

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
            waveform, sr = torchaudio.load(filepath)
            
            # Преобразование в моно и нужную частоту дискретизации
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)

            # Вырезаем случайный сегмент
            if waveform.size(1) > self.segment_len:
                start = random.randint(0, waveform.size(1) - self.segment_len)
                waveform = waveform[:, start:start+self.segment_len]
            
            return waveform.squeeze(0)
        except Exception as e:
            print(f"Ошибка при загрузке файла {filepath}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1)) # Пробуем другой файл

def collate_fn_pretrain(batch):
    waveforms = [item for item in batch]
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
            
    masked_features = cnn_features.clone()
    masked_features.transpose(1, 2)[mask] = 0 # Зануляем замаскированные векторы
    
    return masked_features, mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # 1. Данные
    dataset = UnsupervisedAudioDataset(Path(args.dataset_path))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            collate_fn=collate_fn_pretrain, num_workers=4)

    # 2. Модель
    model = CustomSpeechEncoder(
        d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layers
    ).to(device)

    # 3. Функция потерь и оптимизатор
    criterion = ContrastiveLoss(temperature=args.temperature).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 4. Цикл обучения
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, waveforms in enumerate(dataloader):
            waveforms = waveforms.to(device)
            
            optimizer.zero_grad()
            
            # Прямой проход
            _, cnn_features = model(waveforms) # Нам нужны оба выхода
            
            # Маскирование
            masked_cnn_features, mask = apply_mask(cnn_features)
            
            # Прогоняем замаскированные признаки через трансформер, чтобы получить предсказания
            # Для этого нужно немного изменить модель или сделать второй проход.
            # Для чистоты архитектуры - делаем второй проход.
            # Эффективнее было бы изменить forward модели, чтобы она принимала cnn_features.
            
            # --- ВАРИАНТ 1: Второй проход (проще, но медленнее)
            # Для этого надо переделать модель, чтобы она могла принимать фичи на вход
            # Давайте для простоты предположим, что мы можем это сделать
            
            # --- ВАРИАНТ 2: Изменяем логику тут (практичнее для этого скрипта)
            projected = model.input_projection(masked_cnn_features.transpose(1, 2))
            pos_encoded = projected + model.positional_encoding[:, :projected.size(1), :]
            transformer_output = model.transformer_encoder(pos_encoded) # -> [B, T_reduced, d_model]
            
            # Выбираем выходы и цели только для замаскированных позиций
            transformer_masked_outputs = transformer_output[mask.transpose(0, 1)]
            cnn_masked_targets = cnn_features.transpose(1, 2)[mask.transpose(0, 1)]

            if transformer_masked_outputs.size(0) == 0: continue # если ничего не замаскировалось

            # Нужно reshape, чтобы было [Batch_effective, 1, Dim]
            # Это усложнение. Давайте упростим: будем сравнивать все выходы
            transformer_masked_outputs = transformer_output.view(-1, args.d_model)[mask.flatten()]
            cnn_masked_targets = cnn_features.transpose(1, 2).reshape(-1, 512)[mask.flatten()]
            # Для cnn_masked_targets нужна проекция в d_model
            projected_targets = model.input_projection(cnn_masked_targets)

            loss = criterion(transformer_masked_outputs.unsqueeze(0), projected_targets.unsqueeze(0))

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f"Эпоха [{epoch+1}/{args.epochs}], Шаг [{i+1}/{len(dataloader)}], Потери: {loss.item():.4f}")

        print(f"Средние потери за эпоху {epoch+1}: {total_loss / len(dataloader):.4f}")

        # Сохранение чекпоинта
        torch.save(model.state_dict(), f"{args.save_path}/pretrained_encoder_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт предобучения модели ASR")
    parser.add_argument('--dataset_path', type=str, required=True, help='Путь к корневой папке с аудиофайлами')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Папка для сохранения чекпоинтов')
    
    # Гиперпараметры модели
    parser.add_argument('--d_model', type=int, default=768, help='Размерность модели трансформера')
    parser.add_argument('--n_head', type=int, default=12, help='Количество голов во внимании')
    parser.add_argument('--n_layers', type=int, default=12, help='Количество слоев трансформера')

    # Гиперпараметры обучения
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=8, help='Размер батча')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Скорость обучения')
    parser.add_argument('--temperature', type=float, default=0.1, help='Температура для Contrastive Loss')

    args = parser.parse_args()
    Path(args.save_path).mkdir(exist_ok=True, parents=True)
    main(args)