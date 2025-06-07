import torch
import torch.nn as nn
import torch.nn.functional as F




class SmallModel(nn.Module):
    def __init__(self, num_classes=19632, num_blocks=5, embedding_dim=1100, channels=256):
        super().__init__()
        

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
