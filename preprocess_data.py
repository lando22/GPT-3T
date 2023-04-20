from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, Dataset
from train_model import TextDataset

# Train a BPE tokenizer on the dataset
def train_bpe_tokenizer(file_path, vocab_size=52000):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>"])
    tokenizer.train(files=[file_path], trainer=trainer)
    return tokenizer
  
def preprocess_data(datafile, sequence_length=128, batch_size=55, vocab_size=52000):
    tokenizer = train_bpe_tokenizer(datafile, vocab_size)
    dataset = TextDataset(datafile, tokenizer, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, tokenizer
