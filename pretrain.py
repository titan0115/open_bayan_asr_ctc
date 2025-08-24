import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.amp as amp
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import torchaudio
import random
import librosa
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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

# TensorBoard
TENSORBOARD_LOG_DIR = "./runs/pretrain"
# =========================

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3'}

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
        for f in tqdm(dataset.files, desc="Получение длин файлов"):
            try:
                info = torchaudio.info(str(f))
                self.lengths.append(info.num_frames)
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


class UnsupervisedAudioDataset(Dataset):
    def __init__(self, root_dir: Path, segment_duration_sec=5):
        self.files = [p for p in root_dir.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS]
        self.target_sr = 16000
        self.segment_len = self.target_sr * segment_duration_sec
        
        print(f"Найдено {len(self.files)} аудиофайлов")

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
    
    # НОВОЕ: Логируем статистику длин для диагностики эффективности сортировки
    lengths = [len(w) for w in waveforms]
    if hasattr(collate_fn_pretrain, 'batch_count'):
        collate_fn_pretrain.batch_count += 1
    else:
        collate_fn_pretrain.batch_count = 1
    
    # Логируем статистику каждые 100 батчей
    if collate_fn_pretrain.batch_count % 100 == 0:
        print(f"Батч {collate_fn_pretrain.batch_count}: длины {min(lengths)}-{max(lengths)}, "
              f"средняя {sum(lengths)/len(lengths):.0f}, "
              f"паддинг {(max(lengths) - min(lengths)) / max(lengths) * 100:.1f}%")
    
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

    # Создаем датасет
    dataset = UnsupervisedAudioDataset(Path(DATASET_PATH))
    
    # НОВОЕ: Используем LengthGroupedSampler для правильной группировки
    sampler = LengthGroupedSampler(dataset, batch_size=BATCH_SIZE, shuffle=True)

    dataloader = DataLoader(dataset, 
                            sampler=sampler,  # Используем sampler, а не batch_sampler
                            batch_size=BATCH_SIZE,  # Возвращаем batch_size
                            shuffle=False,  # shuffle=False, так как сэмплер сам занимается перемешиванием
                            collate_fn=collate_fn_pretrain, 
                            num_workers=NUM_WORKERS, 
                            pin_memory=PIN_MEMORY,
                            prefetch_factor=PREFETCH_FACTOR,
                            persistent_workers=PERSISTENT_WORKERS)

    model = CustomSpeechEncoder(d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS).to(device)

    criterion = ContrastiveLoss(temperature=TEMPERATURE).to(device)
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

    # Инициализация TensorBoard
    writer = SummaryWriter(TENSORBOARD_LOG_DIR)
    
    # Записываем гиперпараметры
    hparams = {
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layers': N_LAYERS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'temperature': TEMPERATURE,
        'epochs': EPOCHS,
        'use_amp': USE_AMP,
        'num_workers': NUM_WORKERS
    }
    writer.add_hparams(hparams, {'hparam/accuracy': 0.0})
    
    global_step = 0

    # Общий прогресс-бар для всех эпох
    epoch_pbar = tqdm(range(EPOCHS), desc="Обучение", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        
        # Создаем прогресс-бар для текущей эпохи
        pbar = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{EPOCHS}", 
                   leave=False, unit="it")
        
        for i, waveforms in enumerate(pbar):
            if waveforms is None:
                continue
                
            waveforms = waveforms.to(device)
            
            # НОВОЕ: Проверка данных на входе для диагностики
            if i == 0 and epoch == 0:  # Только для первого батча первой эпохи
                print(f"Диагностика данных:")
                print(f"  Waveform shape: {waveforms.shape}")
                print(f"  Waveform min/max: {waveforms.min():.4f}/{waveforms.max():.4f}")
                print(f"  Waveform mean/std: {waveforms.mean():.4f}/{waveforms.std():.4f}")
                print(f"  Non-zero elements: {(waveforms != 0).sum()}/{waveforms.numel()}")
            
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

        Path(SAVE_PATH).mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), f"{SAVE_PATH}/pretrained_encoder_epoch_{epoch+1}.pt")

    # Закрываем TensorBoard writer
    writer.close()
    print(f"TensorBoard логи сохранены в {TENSORBOARD_LOG_DIR}")

if __name__ == "__main__":
    main()