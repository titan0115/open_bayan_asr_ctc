# 2_loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Контрастивная функция потерь для self-supervised предобучения.
    Задача: предсказание трансформера для замаскированного шага должно быть похоже 
    на "истинный" выход CNN для этого шага и не похоже на выходы для других шагов.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, transformer_outputs, cnn_targets, num_negatives=100):
        """
        Args:
            transformer_outputs (Tensor): Выход трансформера для замаскированных позиций [Batch, Num_Masked, Dim]
            cnn_targets (Tensor): "Истинные" векторы из CNN для тех же позиций [Batch, Num_Masked, Dim]
        """
        batch_size, num_masked, dim = transformer_outputs.shape
        
        # Выпрямляем тензоры для удобства
        preds = transformer_outputs.view(-1, dim) # [B * Num_Masked, Dim]
        targets = cnn_targets.view(-1, dim) # [B * Num_Masked, Dim]

        # Вычисляем косинусное сходство для позитивных пар (предсказание и его истинная цель)
        positive_similarity = F.cosine_similarity(preds, targets, dim=-1) # -> [B * Num_Masked]

        # Генерируем негативные примеры
        # Для простоты берем случайные векторы из самого `cnn_targets`, но не совпадающие с истинным
        total_samples = targets.size(0)
        negative_samples = []
        for i in range(total_samples):
            # Создаем маску, чтобы не выбрать самого себя
            mask = torch.ones(total_samples, dtype=torch.bool, device=preds.device)
            mask[i] = False
            # Выбираем `num_negatives` случайных индексов
            neg_indices = torch.multinomial(mask.float(), num_negatives, replacement=True)
            negative_samples.append(targets[neg_indices])
        
        negatives = torch.stack(negative_samples) # -> [B*Num_Masked, Num_Negatives, Dim]

        # Вычисляем косинусное сходство для негативных пар
        # Правильный способ: вычисляем сходство между preds и каждым негативным примером
        negative_similarity = F.cosine_similarity(preds.unsqueeze(1), negatives, dim=2)

        # Объединяем позитивное и негативное сходство
        # [B*Num_Masked, 1 + Num_Negatives]
        logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1)
        
        # Применяем температуру
        logits /= self.temperature
        
        # Целевые метки для CrossEntropyLoss - всегда 0, так как позитивный пример всегда первый
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=preds.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss