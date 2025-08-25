# 4_finetune.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.amp as amp
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import torchaudio
import pandas as pd
import json
import random
import numpy as np
import librosa
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Импортируем наши модули
from model import CustomSpeechEncoder, CTClassificationHead, ASRModel

# ===== ГИПЕРПАРАМЕТРЫ =====
# Пути
DATASET_PATH = "./data"  # Путь к папке с данными в формате JSON
PRETRAINED_CHECKPOINT = "./checkpoints/pretrained_encoder_epoch_9.pt"  # Путь к файлу с весами предобученного энкодера
SAVE_PATH = "./finetuned_model"  # Папка для сохранения финальной модели

# Гиперпараметры обучения
EPOCHS = 100  # Количество эпох
BATCH_SIZE = 16  # Размер батча
LEARNING_RATE = 3e-4  # Скорость обучения
USE_AMP = True  # Использовать автоматическое смешанное представление
USE_CHECKPOINTING = True  # Использовать gradient checkpointing
SEED = 42  # Сид для воспроизводимости
MAX_AUDIO_DURATION_SEC = 30 # <--- НОВОЕ: Максимальная длина аудио в секундах

# Параметры загрузки данных
NUM_WORKERS = 4  # Количество воркеров для загрузки данных
PIN_MEMORY = True  # Использовать закрепленную память для ускорения передачи на GPU
PREFETCH_FACTOR = 2  # Количество батчей, предзагружаемых каждым воркером
PERSISTENT_WORKERS = True  # Сохранять воркеров между эпохами

# TensorBoard
TENSORBOARD_LOG_DIR = f"./runs/finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# =========================

# Устанавливаем сиды для воспроизводимости
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Определяем словарь (важно, чтобы 0 был BLANK символом для CTC)
VOCAB = ["_"] + list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя") + [" "]
char_to_int = {char: i for i, char in enumerate(VOCAB)}
int_to_char = {i: char for i, char in enumerate(VOCAB)}


class LengthGroupedSampler(Sampler):
    """
    Сэмплер для группировки файлов по длинам для минимизации паддинга
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Получаем длины всех файлов один раз
        print("Анализ длин файлов для сэмплера...")
        self.lengths = []
        for item in tqdm(dataset.data, desc="Получение длин файлов"):
            try:
                # Используем duration из JSON или оцениваем по длине текста
                duration = item.get('duration', len(item['text']) * 0.1)  # Примерная оценка
                self.lengths.append(int(duration * 16000))  # Конвертируем в сэмплы
            except Exception:
                self.lengths.append(0)  # Пометим проблемные файлы
        
        # Создаем группы (бакеты) индексов
        self.buckets = self._create_buckets()

    def _create_buckets(self):
        # Сортируем индексы по длинам, исключая проблемные файлы
        valid_indices = [i for i, length in enumerate(self.lengths) if length > 0]
        sorted_indices = sorted(valid_indices, key=lambda i: self.lengths[i])
        
        buckets = []
        # Разбиваем отсортированные индексы на батчи
        for i in range(0, len(sorted_indices), self.batch_size):
            bucket = sorted_indices[i:i+self.batch_size]
            if len(bucket) == self.batch_size:  # Только полные батчи
                buckets.append(bucket)
            
        print(f"Создано {len(buckets)} батчей с группировкой по длинам")
        return buckets

    def __iter__(self):
        # Перемешиваем порядок бакетов, если нужно
        if self.shuffle:
            random.shuffle(self.buckets)
        
        # Проходим по бакетам и выдаем индексы
        for bucket in self.buckets:
            # Можно дополнительно перемешать элементы внутри бакета
            if self.shuffle:
                random.shuffle(bucket)
            yield from bucket

    def __len__(self):
        # Возвращаем общее количество элементов, а не количество батчей
        return len(self.buckets) * self.batch_size


class SupervisedAudioDataset(Dataset):
    def __init__(self, data_dir: Path, max_duration_sec: int): # <--- НОВОЕ: Добавляем max_duration_sec
        self.data = []
        self.target_sr = 16000
        self.max_samples = self.target_sr * max_duration_sec # <--- НОВОЕ: Максимальное кол-во сэмплов
        
        # Собираем все JSON файлы из папки data
        json_files = list(data_dir.rglob('*.json'))
        print(f"Найдено {len(json_files)} JSON файлов")
        
        skipped_long_files = 0
        
        for json_file in tqdm(json_files, desc="Загрузка и фильтрация метаданных"):
            try:
                # Ищем соответствующий аудиофайл
                audio_path = None
                for audio_ext in ['.wav', '.mp3', '.flac']:
                    potential_path = json_file.parent / f"{json_file.stem}{audio_ext}"
                    if potential_path.exists():
                        audio_path = potential_path
                        break
                
                if audio_path:
                    # НОВОЕ: Проверяем длительность аудио БЕЗ полной загрузки
                    info = torchaudio.info(str(audio_path))
                    
                    # Проверяем, что аудиофайл не превышает лимит
                    if info.num_frames <= self.max_samples:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            item = json.load(f)
                        
                        item['audio_path'] = str(audio_path)
                        # Добавляем точную длительность для сэмплера
                        item['duration'] = info.num_frames / info.sample_rate 
                        self.data.append(item)
                    else:
                        skipped_long_files += 1

            except Exception as e:
                print(f"Ошибка при обработке {json_file}: {e}")
        
        print(f"Загружено {len(self.data)} записей.")
        print(f"Пропущено {skipped_long_files} слишком длинных файлов (>{max_duration_sec} сек).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        filepath = Path(item['audio_path'])
        text = item['text'].lower()

        try:
            try:
                waveform, sr = torchaudio.load(filepath)
                if waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != self.target_sr:
                    waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
            except Exception:
                # Fallback на librosa для проблемных файлов
                waveform, sr = librosa.load(str(filepath), sr=self.target_sr, mono=True)
                waveform = torch.from_numpy(waveform).unsqueeze(0)
        except Exception as e:
            print(f"Ошибка при загрузке аудио {filepath}: {e}")
            # Возвращаем пустой тензор в случае ошибки
            waveform = torch.zeros(1, 1) # Возвращаем очень короткий тензор, который будет отфильтрован в collate_fn
        
        labels = [char_to_int.get(c, char_to_int['_']) for c in text]
        
        return waveform.squeeze(0), torch.tensor(labels, dtype=torch.long)

def collate_fn_finetune(batch):
    waveforms, labels = zip(*batch)
    
    # Фильтруем пустые записи
    valid_pairs = [(w, l) for w, l in zip(waveforms, labels) if torch.sum(torch.abs(w)) > 0]
    if not valid_pairs:
        return None, None, None, None
    
    waveforms, labels = zip(*valid_pairs)
    
    # НОВОЕ: Логируем статистику длин для диагностики эффективности сортировки
    lengths = [len(w) for w in waveforms]
    if hasattr(collate_fn_finetune, 'batch_count'):
        collate_fn_finetune.batch_count += 1
    else:
        collate_fn_finetune.batch_count = 1
    
    # Логируем статистику каждые 100 батчей
    if collate_fn_finetune.batch_count % 100 == 0:
        print(f"Батч {collate_fn_finetune.batch_count}: длины {min(lengths)}-{max(lengths)}, "
              f"средняя {sum(lengths)/len(lengths):.0f}, "
              f"паддинг {(max(lengths) - min(lengths)) / max(lengths) * 100:.1f}%")
    
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    waveform_lengths = torch.tensor([w.size(0) for w in waveforms])
    
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    label_lengths = torch.tensor([len(l) for l in labels])
    
    return padded_waveforms, waveform_lengths, padded_labels, label_lengths

def main():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Используется устройство: {device}")

    # 1. Данные
    dataset = SupervisedAudioDataset(Path(DATASET_PATH), max_duration_sec=MAX_AUDIO_DURATION_SEC)
    
    # НОВОЕ: Используем LengthGroupedSampler для правильной группировки
    sampler = LengthGroupedSampler(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    dataloader = DataLoader(dataset, 
                            sampler=sampler,  # Используем sampler, а не batch_sampler
                            batch_size=BATCH_SIZE,  # Возвращаем batch_size
                            shuffle=False,  # shuffle=False, так как сэмплер сам занимается перемешиванием
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
    
    # НОВОЕ: Добавляем планировщик скорости обучения с разогревом
    num_training_steps = len(dataloader) * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% шагов на разогрев

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0  # После разогрева LR будет управляться оптимизатором

    scheduler = LambdaLR(optimizer, lr_lambda)
    
    scaler = amp.GradScaler(enabled=(USE_AMP and device.type == 'cuda'))

    # Создаем папку для логов TensorBoard
    Path(TENSORBOARD_LOG_DIR).mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard логи будут сохранены в: {TENSORBOARD_LOG_DIR}")
    
    # Инициализация TensorBoard
    writer = SummaryWriter(TENSORBOARD_LOG_DIR)
    
    # Записываем гиперпараметры
    hparams = {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'num_workers': NUM_WORKERS,
        'vocab_size': len(VOCAB),
        'use_amp': USE_AMP,
        'use_checkpointing': USE_CHECKPOINTING
    }
    
    global_step = 0

    # 4. Цикл обучения
    # Общий прогресс-бар для всех эпох
    epoch_pbar = tqdm(range(EPOCHS), desc="Дообучение", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        
        # Создаем прогресс-бар для текущей эпохи
        pbar = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{EPOCHS}", 
                   leave=False, unit="it")
        
        for i, batch_data in enumerate(pbar):
            if batch_data[0] is None:  # Пропускаем пустые батчи
                continue
                
            waveforms, wave_lens, labels, label_lens = batch_data
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            # НОВОЕ: Проверка данных на входе для диагностики
            if i == 0 and epoch == 0:  # Только для первого батча первой эпохи
                print(f"Диагностика данных:")
                print(f"  Waveform shape: {waveforms.shape}")
                print(f"  Waveform min/max: {waveforms.min():.4f}/{waveforms.max():.4f}")
                print(f"  Waveform mean/std: {waveforms.mean():.4f}/{waveforms.std():.4f}")
                print(f"  Non-zero elements: {(waveforms != 0).sum()}/{waveforms.numel()}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Label lengths: {label_lens}")
            
            optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast(device_type=device_str, dtype=torch.float16, enabled=(USE_AMP and device.type == 'cuda')):
                logits = model(waveforms, use_checkpointing=USE_CHECKPOINTING) # -> [B, T, C]
                
                # Подготовка для CTC Loss
                log_probs = nn.functional.log_softmax(logits, dim=2)
                log_probs = log_probs.transpose(0, 1) # -> [T, B, C]
                
                # Длины входов для CTC должны соответствовать выходу модели
                # Упрощенный расчет - в реальности нужно учитывать все страйды
                input_lengths = torch.div(wave_lens, 320, rounding_mode='floor')  # Примерный коэффициент сжатия
                input_lengths = torch.clamp(input_lengths, min=1)  # Минимум 1
                
                loss = criterion(log_probs, labels, input_lengths, label_lens)

            scaler.scale(loss).backward()
            
            # НОВОЕ: Клиппинг градиентов для стабилизации обучения
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # НОВОЕ: Делаем шаг планировщика
            scheduler.step()
            
            total_loss += loss.item()
            
            # Обновляем прогресс-бар с текущими потерями
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (i+1):.4f}'
            })
            
            # Записываем в TensorBoard
            writer.add_scalar('Loss/Batch', loss.item(), global_step)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)
            global_step += 1

        avg_loss = total_loss / len(dataloader)
        
        # Записываем средние потери за эпоху в TensorBoard
        writer.add_scalar('Loss/Epoch', avg_loss, epoch + 1)
        
        # Обновляем общий прогресс-бар
        epoch_pbar.set_postfix({
            'Epoch': f'{epoch+1}/{EPOCHS}',
            'Avg Loss': f'{avg_loss:.4f}'
        })

        # Сохранение чекпоинта каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            Path(SAVE_PATH).mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), f"{SAVE_PATH}/finetuned_asr_model_epoch_{epoch+1}.pt")

        # НОВОЕ: Добавляем явную очистку в конце эпохи
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Сохранение финальной модели
    Path(SAVE_PATH).mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), f"{SAVE_PATH}/finetuned_asr_model.pt")
    print(f"Финальная модель сохранена в {SAVE_PATH}/finetuned_asr_model.pt")

    # Записываем финальную среднюю потерю в гиперпараметры
    final_avg_loss = total_loss / len(dataloader)
    writer.add_hparams(hparams, {'hparam/final_avg_loss': final_avg_loss})
    
    # Закрываем TensorBoard writer
    writer.close()
    print(f"TensorBoard логи сохранены в {TENSORBOARD_LOG_DIR}")

if __name__ == "__main__":
    main()