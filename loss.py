# 2_loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Контрастивная функция потерь для self-supervised предобучения.
    Эффективная векторизованная реализация.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, transformer_outputs, cnn_targets):
        """
        Args:
            transformer_outputs (Tensor): Выход трансформера для замаскированных позиций [Batch, Num_Masked, Dim]
            cnn_targets (Tensor): "Истинные" векторы из CNN для тех же позиций [Batch, Num_Masked, Dim]
        """
        # 1. Выпрямляем тензоры и нормализуем их для косинусного сходства
        # Это делает вычисление более стабильным и эквивалентно F.cosine_similarity
        preds = transformer_outputs.contiguous().view(-1, transformer_outputs.size(-1))
        preds = F.normalize(preds, p=2, dim=1) # [N, Dim], где N = B * Num_Masked

        targets = cnn_targets.contiguous().view(-1, cnn_targets.size(-1))
        targets = F.normalize(targets, p=2, dim=1) # [N, Dim]

        # 2. Вычисляем матрицу попарных сходств
        # Результат - матрица [N, N], где sim_matrix[i, j] - сходство между i-м предсказанием и j-м таргетом
        similarity_matrix = torch.matmul(preds, targets.T)

        # 3. Находим позитивные и негативные примеры
        # Позитивные примеры находятся на диагонали матрицы.
        # Все остальные - негативные.
        
        # Создаем маску для позитивных примеров (диагональ)
        num_samples = preds.size(0)
        device = preds.device
        
        # logits для InfoNCE loss
        # Нам нужно сравнить каждое предсказание (строка) с его позитивным таргетом (на диагонали)
        # и всеми остальными (негативными) таргетами в этой же строке.
        
        # Метки для CrossEntropyLoss - это просто индексы диагональных элементов
        labels = torch.arange(num_samples, device=device)

        # Применяем температуру
        logits = similarity_matrix / self.temperature
        
        # 4. Вычисляем CrossEntropyLoss
        # Для каждого предсказания `preds[i]` мы хотим, чтобы `logits[i, i]` был максимальным.
        loss = F.cross_entropy(logits, labels)
        
        return loss