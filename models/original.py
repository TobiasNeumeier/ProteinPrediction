import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class OriginalModel(nn.Module):
    def __init__(self, num_classes=19632, num_blocks=5, embedding_dim=1100, channels=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_layer = nn.Conv1d(21, channels, kernel_size=3, padding=1)  # 21 for amino acids + gap/mask


        self.res_blocks = nn.Sequential(*[
            ResidualBlock(channels, dilation=2**i) for i in range(num_blocks)
        ])

        # Final 1x1 conv projects to embedding dimension
        self.to_embedding = nn.Conv1d(channels, embedding_dim, kernel_size=1)

        # Final classifier projects embedding to Pfam classes
        self.classifier = nn.Linear(embedding_dim, num_classes)


    def forward(self, x_onehot):
        """
        Args:
            x_onehot: Tensor of shape [batch_size, seq_len, 21] (one-hot encoded input)
        Returns:
            logits: [batch_size, seq_len, num_classes]
            embeddings: [batch_size, seq_len, embedding_dim]
        """
        x = x_onehot.permute(0, 2, 1)  # → [batch_size, 21, seq_len]
        x = self.input_layer(x)        # → [batch_size, channels, seq_len]
        emb = self.to_embedding(x)    # → [batch_size, 1100, seq_len]
        emb = emb.permute(0, 2, 1)    # → [batch_size, seq_len, 1100]
        logits = self.classifier(emb) # → [batch_size, seq_len, num_classes]
        return logits, emb
