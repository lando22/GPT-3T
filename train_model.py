import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm
import time
from torch.cuda.amp import GradScaler, autocast

# Updated GPT model with layer normalization and dropout
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len=5000, dropout=0.1):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.embedding(x) * math.sqrt(self.d_model) + self.pos_encoder(pos)
        x = self.dropout(x)
        x = self.transformer_decoder(x, torch.zeros_like(x))
        x = self.norm(x)
        x = self.fc(x)
        return x

class TextDataset(Dataset):
    """Dataset that generates samples on-the-fly to reduce memory usage."""

    def __init__(self, file_path, tokenizer, sequence_length):
        self.sequence_length = sequence_length
        tokens = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                tokens.extend(tokenizer.encode(line).ids)

        self.tokens = torch.tensor(tokens, dtype=torch.long)
        # Number of non-overlapping sequences
        self.num_samples = max(
            0,
            (len(self.tokens) - (sequence_length + 3)) // sequence_length,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        input_seq = self.tokens[start : start + self.sequence_length]
        target_seq = self.tokens[
            start + self.sequence_length : start + self.sequence_length + 3
        ]
        return input_seq, target_seq

      
      
def train_model(
    vocab_size,
    d_model,
    nhead,
    num_layers,
    dataloader,
    dropout=0.1,
    epochs=5,
    learning_rate=0.00004,
    grad_clip=1.0,
    use_amp=True,
):
    model = GPT(vocab_size, d_model, nhead, num_layers, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        total_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        model.train()
        for input_seq, target_seq in progress_bar:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                output = model(input_seq)
                output = output[:, -3:, :]
                loss = criterion(
                    output.reshape(-1, vocab_size), target_seq.reshape(-1)
                )
                loss = loss.mean()

            scaler.scale(loss).backward()

            # Apply gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Update learning rate scheduler

            total_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": loss.item()})

        avg_train_loss = total_loss / total_batches
        elapsed_time = time.time() - start_time

        # Print average train loss and elapsed time for the current epoch
        print(
            f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_train_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s"
        )
