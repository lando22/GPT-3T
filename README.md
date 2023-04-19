# GPT-3T

**This repo is not fully complete. The tokenizing functions could be much improved for memory efficiency and training could be optimized**

GPT-3T is an experiment designed to see what happens if you force a GPT model to predict further into the future by predicting 3 tokens at a time, instead of just one. This model is built using PyTorch and the tokenizers library.

My reason for building GPT-3T is I suspect that training large language models to predict further into the future could potentially lead to models that can do better multistep reasoning because the pretraining objective forces them to "think further ahead".

I want to disclose that I could be incredibly off in my thinking, but the experiement is worth trying. At a small scale, GPT-3T shows *some* promise, but is severly limited so far because of its very minimal data and model size.

## Overview

This project provides two main components:

1. `train_model.py`: Contains the GPT model architecture and the training loop.
2. `preprocess_data.py`: Provides functions for training a BPE tokenizer on the dataset and preprocessing the data for the GPT-3T model.

## Usage

To use this project, follow these steps:

1. Install the required dependencies:

```bash
pip install torch tokenizers tqdm
```

2. Prepare your text dataset as a `.txt` file. This can be a simple raw text file without any special formatting.

3. In your main script, import and use the `preprocess_data` and `train_model` functions for highly simplified training to build a GPT-3T model:


```python
from train_model import train_model
from preprocess_data import preprocess_data

datafile = "<datafile.txt>"

dataloader, tokenizer = preprocess_data(datafile)
train_model(
    vocab_size=52000,
    d_model=512,
    nhead=8,
    num_layers=8,
    dropout=0.1,
    dataloader=dataloader
)

```

Replace `<datafile.txt>` with the path to your dataset file.

## Customization
You can customize the model and training parameters by modifying the default values in the train_model() and preprocess_data() functions:

* sequence_length: The length of the input sequences for the model.
* batch_size: The number of samples in each training batch.
* vocab_size: The size of the vocabulary used by the BPE tokenizer.
* epochs: The number of training epochs.
* learning_rate: The learning rate for the optimizer.
* grad_clip: The gradient clipping value.

## Summarized Experiments

Limited experiments with GPT-3T have shown mixed results. The model does not perform exceptionally well when compared to traditional GPT models. However, it is important to note that the experiments have been incredibly limited in scope, as I have not tested GPT-3T with larger datasets or much larger models.

The largest dataset used in my experiments was Webtext-10K (from huggingface), and the largest model had appox. 300 million parameters. More extensive experimentation may be necessary to fully understand the potential of GPT-3T in predicting further into the future with higher accuracy.

I would encourage the community to test the GPT-3T model with much larger datasets and bigger models, as well as explore different hyperparameter settings to improve its performance. Unfortunately, my personal compute resources are not incredibly high which has limited this project.


## Why 3 tokens?
There is no particular reasoning behind this choice other than it seems to be a baby step into trying something different!
