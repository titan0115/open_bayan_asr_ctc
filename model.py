# 1_model_architecture.py
import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    """
    Сверточный извлекатель признаков (Feature Extractor).
    Принимает сырую аудиоволну и преобразует ее в последовательность локальных признаков,
    уменьшая временную размерность.
    """
    def __init__(self, in_channels=1, output_dim=512):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, 512, kernel_size=10, stride=5),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(512, output_dim, kernel_size=2, stride=2),
        ])

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1) # [B, T] -> [B, 1, T]

        for layer in self.conv_layers:
            waveform = layer(waveform)
        
        # Выход: [B, output_dim, T_reduced]
        return waveform

class CustomSpeechEncoder(nn.Module):
    """
    Основной энкодер, объединяющий CNN и Трансформер.
    Его задача - создать финальные контекстуально-обогащенные эмбеддинги аудио.
    """
    def __init__(self, cnn_output_dim=512, d_model=768, n_head=12, n_layers=12, dim_feedforward=3072, dropout=0.1):
        super().__init__()
        
        self.feature_extractor = CNNFeatureExtractor(output_dim=cnn_output_dim)
        
        self.input_projection = nn.Linear(cnn_output_dim, d_model)
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, 2048, d_model)) # 2048 - макс. длина
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, waveform: torch.Tensor, padding_mask: torch.Tensor = None):
        # 1. Извлекаем признаки из аудиоволны
        cnn_features = self.feature_extractor(waveform) # -> [B, C, T_reduced]
        
        # 2. Подготавливаем для трансформера
        x = cnn_features.transpose(1, 2) # -> [B, T_reduced, C]
        x = self.input_projection(x) # -> [B, T_reduced, d_model]
        
        # 3. Добавляем позиционную информацию
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # 4. Прогоняем через Трансформер
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)
        
        return x, cnn_features

class CTClassificationHead(nn.Module):
    """
    "Голова" для классификации, которая добавляется поверх энкодера для fine-tuning.
    Предсказывает символы для CTC Loss.
    """
    def __init__(self, input_dim=768, vocab_size=33): # 32 буквы + blank
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.activation = nn.GELU()
        self.output_layer = nn.Linear(input_dim, vocab_size)

    def forward(self, embeddings):
        x = self.dense(embeddings)
        x = self.activation(x)
        logits = self.output_layer(x)
        return logits

class ASRModel(nn.Module):
    """
    Полная модель для распознавания речи (Энкодер + Голова).
    Используется на этапе fine-tuning.
    """
    def __init__(self, encoder: CustomSpeechEncoder, head: CTClassificationHead):
        super().__init__()
        self.encoder = encoder
        self.head = head
        
    def forward(self, waveform: torch.Tensor, padding_mask: torch.Tensor = None):
        embeddings, _ = self.encoder(waveform, padding_mask)
        logits = self.head(embeddings)
        return logits