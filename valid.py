import torch
import torch.nn as nn
from pathlib import Path
import torchaudio
import json
import random
import numpy as np
import librosa
from tqdm import tqdm

# Импортируем модель и компоненты
from model import CustomSpeechEncoder, CTClassificationHead, ASRModel

# ===== КОНФИГУРАЦИЯ =====
DATASET_PATH = "./data"
MODEL_PATH = "./finetuned_model/finetuned_asr_model_epoch_10.pt"
NUM_SAMPLES = 5  # Количество примеров для валидации
SEED = 42

# Словарь символов (должен совпадать с finetune.py)
VOCAB = ["_"] + list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя") + [" "]
char_to_int = {char: i for i, char in enumerate(VOCAB)}
int_to_char = {i: char for i, char in enumerate(VOCAB)}

# Устанавливаем сиды для воспроизводимости
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class ValidationDataset:
    """Упрощенный датасет для валидации"""
    def __init__(self, data_dir: Path, max_samples: int = 100):
        self.data = []
        self.target_sr = 16000

        # Собираем все JSON файлы из папки data
        json_files = list(data_dir.rglob('*.json'))[:max_samples]  # Берем только первые max_samples файлов
        print(f"Найдено {len(json_files)} JSON файлов (используем первые {max_samples})")

        for json_file in tqdm(json_files, desc="Загрузка метаданных для валидации"):
            try:
                # Ищем соответствующий аудиофайл
                audio_path = None
                for audio_ext in ['.wav', '.mp3', '.flac']:
                    potential_path = json_file.parent / f"{json_file.stem}{audio_ext}"
                    if potential_path.exists():
                        audio_path = potential_path
                        break

                if audio_path:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        item = json.load(f)

                    item['audio_path'] = str(audio_path)
                    self.data.append(item)

            except Exception as e:
                print(f"Ошибка при обработке {json_file}: {e}")

        print(f"Загружено {len(self.data)} записей для валидации.")

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
            waveform = torch.zeros(1, 1)

        return waveform.squeeze(0), text


def ctc_decode(logits, blank_idx=0):
    """Декодирование CTC выхода в текст"""
    # Применяем softmax и берем argmax
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    # CTC декодирование: удаляем повторяющиеся символы и blank
    decoded = []
    prev_char = blank_idx

    for t in range(preds.size(1)):  # Проходим по временным шагам
        char_idx = preds[0, t].item()  # Берем первый (и единственный) элемент батча

        if char_idx != blank_idx and char_idx != prev_char:
            decoded.append(int_to_char[char_idx])

        prev_char = char_idx

    return ''.join(decoded).strip()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # 1. Загружаем модель
    print("Загрузка модели...")
    encoder = CustomSpeechEncoder()
    head = CTClassificationHead(vocab_size=len(VOCAB))
    model = ASRModel(encoder, head)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"Модель загружена из {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Ошибка: файл модели не найден по пути {MODEL_PATH}")
        return
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return

    # 2. Создаем датасет
    dataset = ValidationDataset(Path(DATASET_PATH), max_samples=50)  # Берем больше файлов, чтобы выбрать случайные

    if len(dataset) < NUM_SAMPLES:
        print(f"Предупреждение: в датасете только {len(dataset)} файлов, возьмем все доступные")
        num_samples = len(dataset)
    else:
        num_samples = NUM_SAMPLES

    # Выбираем случайные индексы
    random_indices = random.sample(range(len(dataset)), num_samples)
    print(f"Выбрано {num_samples} случайных примеров для валидации")

    # 3. Валидация
    results = []

    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            print(f"\n--- Пример {i+1}/{num_samples} ---")

            # Получаем данные
            waveform, ground_truth = dataset[idx]
            print(f"Ground truth: '{ground_truth}'")

            # Проверяем, что аудио не пустое
            if torch.sum(torch.abs(waveform)) == 0:
                print("Пропускаем пустой аудиофайл")
                continue

            # Подготавливаем тензор для модели
            waveform = waveform.unsqueeze(0).to(device)  # Добавляем размерность батча

            try:
                # Прогоняем через модель
                logits = model(waveform)

                # Декодируем предсказание
                predicted_text = ctc_decode(logits)

                print(f"Prediction:   '{predicted_text}'")

                # Сравниваем
                # Простая метрика - точное совпадение
                exact_match = predicted_text.lower() == ground_truth.lower()
                print(f"Точное совпадение: {exact_match}")

                # Сохраняем результат
                results.append({
                    'ground_truth': ground_truth,
                    'prediction': predicted_text,
                    'exact_match': exact_match
                })

            except Exception as e:
                print(f"Ошибка при обработке примера {i+1}: {e}")
                continue

    # 4. Итоговая статистика
    if results:
        exact_matches = sum(1 for r in results if r['exact_match'])
        accuracy = exact_matches / len(results)
        print("\n=== РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ===")
        print(f"Обработано примеров: {len(results)}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Точных совпадений: {exact_matches}/{len(results)}")

        # Показываем детали для каждого примера
        print("\nДетали по примерам:")
        for i, result in enumerate(results, 1):
            print(f"{i}. GT: '{result['ground_truth']}' | Pred: '{result['prediction']}' | Match: {result['exact_match']}")
    else:
        print("Не удалось обработать ни одного примера")


if __name__ == "__main__":
    main()
