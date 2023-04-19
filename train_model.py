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
    def __init__(self, file_path, tokenizer, sequence_length):
        self.data = []
        with open(file_path, encoding='utf-8') as f:
            tokens = []
            for line in f:
                encoding = tokenizer.encode(line)
                tokens.extend(encoding.ids)
                while len(tokens) >= sequence_length + 3:
                    input_tensor = torch.tensor(tokens[:sequence_length], dtype=torch.long)
                    target_tensor = torch.tensor(tokens[sequence_length:sequence_length+3], dtype=torch.long)
                    self.data.append((input_tensor, target_tensor))
                    tokens = tokens[sequence_length:]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

      
      
def train_model(vocab_size, d_model, nhead, num_layers, dataloader, dropout=0.1, epochs=10, learning_rate=0.00004, grad_clip=1.0):
  model = GPT(vocab_size, d_model, nhead, num_layers, dropout=dropout)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * epochs)
  criterion = nn.CrossEntropyLoss()

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

          output = model(input_seq)
          output = output[:, -3:, :]

          loss = criterion(output.reshape(-1, vocab_size), target_seq.reshape(-1))
          loss = loss.mean()
          optimizer.zero_grad()
          loss.backward()

          # Apply gradient clipping
          torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

          optimizer.step()
          scheduler.step()  # Update learning rate scheduler

          total_loss += loss.item()
          progress_bar.set_postfix({"Batch Loss": loss.item()})

      avg_train_loss = total_loss / total_batches
      elapsed_time = time.time() - start_time

  # Print average train loss and elapsed time for the current epoch
  print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_train_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s")
