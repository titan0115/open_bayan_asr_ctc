
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CNNFeatureExtractor(nn.Module):
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
            waveform = waveform.unsqueeze(1)
        for layer in self.conv_layers:
            waveform = layer(waveform)
        return waveform


class CustomSpeechEncoder(nn.Module):
    def __init__(self, cnn_output_dim=512, d_model=768, n_head=12, n_layers=12, dim_feedforward=3072, dropout=0.1):
        super().__init__()
        
        # УДАЛЕНО: self.waveform_norm = nn.LayerNorm(normalized_shape=1)
        # LayerNorm не подходит для аудиосигналов переменной длины
        
        self.feature_extractor = CNNFeatureExtractor(output_dim=cnn_output_dim)
        self.input_projection = nn.Linear(cnn_output_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 2048, d_model))
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, waveform: torch.Tensor, use_checkpointing: bool = False):
        """
        Args:
            waveform (Tensor): Входной аудиосигнал.
            use_checkpointing (bool): Флаг для активации gradient checkpointing.
        """
        # НОВОЕ: Правильная нормализация для аудиосигналов переменной длины
        # Убедимся, что у тензора есть размерность для каналов
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [Batch, Length] -> [Batch, 1, Length]

        # Ручная нормализация для каждого аудио в батче
        mean = waveform.mean(dim=-1, keepdim=True)
        std = waveform.std(dim=-1, keepdim=True)
        waveform = (waveform - mean) / (std + 1e-5)  # 1e-5 для стабильности

        cnn_features = self.feature_extractor(waveform)
        
        x = cnn_features.transpose(1, 2)
        x = self.input_projection(x)
        
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        if self.training and use_checkpointing:
            # Применяем checkpoint к каждому слою энкодера
            for layer in self.transformer_encoder.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            x = self.transformer_encoder(x)
            
        x = self.layer_norm(x)
        
        return x, cnn_features

# Классы для fine-tuning, которые пока не используются
class CTClassificationHead(nn.Module):
    def __init__(self, input_dim=768, vocab_size=33):
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
    def __init__(self, encoder: CustomSpeechEncoder, head: CTClassificationHead):
        super().__init__()
        self.encoder = encoder
        self.head = head
        
    def forward(self, waveform: torch.Tensor, use_checkpointing: bool = False):
        # Также добавляем флаг сюда для совместимости
        embeddings, _ = self.encoder(waveform, use_checkpointing=use_checkpointing)
        logits = self.head(embeddings)
        return logits