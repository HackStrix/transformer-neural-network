import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertModel, BertConfig

# Set the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_src = nn.Embedding(vocab_size, 512)
        self.embedding_tgt = nn.Embedding(vocab_size, 512)
        from model import Transformer
        self.transformer = Transformer(Nx=6,
            output_vocab_size=vocab_size
        )        
    def forward(self, src, tgt):
        src_embed = self.embedding_src(src)
        tgt_embed = self.embedding_tgt(tgt)
        output = self.transformer(src_embed, tgt_embed)
        # output = self.fc(output)
        return output, tgt_embed
# Define the Transformer model
# class TransformerModel(nn.Module):
#     def __init__(self, num_tokens):
#         super(TransformerModel, self).__init__()
#         self.config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_tokens)
#         self.encoder = BertModel.from_pretrained('bert-base-uncased', config=self.config)
#         self.decoder = nn.Linear(self.config.hidden_size, num_tokens)

#     def forward(self, encoder_input_ids, decoder_input_ids):
#         encoder_output = self.encoder(encoder_input_ids)[0]
#         decoder_output = self.decoder(encoder_output)
#         return decoder_output

# Define the dataset
import pandas as pd
import torch
from torch.utils.data import Dataset

# Define the special tokens
PAD_TOKEN = 0
UNK_TOKEN = 1
BOS_TOKEN = 2
EOS_TOKEN = 3

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, data, max_len):
        self.encoder_inputs = data['encoder_input'].values
        self.decoder_inputs = data['decoder_input'].values
        self.max_len = max_len
        self.vocab = {'<PAD>': PAD_TOKEN, '<UNK>': UNK_TOKEN, '<BOS>': BOS_TOKEN, '<EOS>': EOS_TOKEN}
        self.reverse_vocab = {PAD_TOKEN: '<PAD>', UNK_TOKEN: '<UNK>', BOS_TOKEN: '<BOS>', EOS_TOKEN: '<EOS>'}
        self.vocab_size = 4

        # Build the vocabulary
        self.build_vocab()

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        encoder_input = self.encoder_inputs[idx]
        decoder_input = self.decoder_inputs[idx]

        # Convert the input sequences to lists of tokens
        encoder_tokens = self.tokenize(encoder_input, is_encoder=True)
        decoder_tokens = self.tokenize(decoder_input, is_encoder=False)

        # Pad the input sequences
        encoder_tokens = self.pad(encoder_tokens, is_encoder=True)
        decoder_tokens = self.pad(decoder_tokens, is_encoder=False)

        # Convert to PyTorch tensors
        encoder_input_ids = torch.tensor(encoder_tokens).unsqueeze(0)
        decoder_input_ids = torch.tensor(decoder_tokens).unsqueeze(0)

        return encoder_input_ids, decoder_input_ids

    def build_vocab(self):
        for sentence in self.encoder_inputs:
            for token in sentence.split():
                if token not in self.vocab:
                    self.vocab[token] = self.vocab_size
                    self.reverse_vocab[self.vocab_size] = token
                    self.vocab_size += 1

        for sentence in self.decoder_inputs:
            for token in sentence.split():
                if token not in self.vocab:
                    self.vocab[token] = self.vocab_size
                    self.reverse_vocab[self.vocab_size] = token
                    self.vocab_size += 1

    def tokenize(self, sentence, is_encoder=True):
        tokens = []
        if is_encoder:
            tokens.append(BOS_TOKEN)
        for token in sentence.split():
            if token in self.vocab:
                tokens.append(self.vocab[token])
            else:
                tokens.append(UNK_TOKEN)
        if not is_encoder:
            tokens.append(EOS_TOKEN)
        return tokens

    def pad(self, tokens, is_encoder=True):
        if is_encoder:
            padded_tokens = [BOS_TOKEN] + tokens + [PAD_TOKEN] * (self.max_len - len(tokens) - 1)
        else:
            padded_tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * (self.max_len - len(tokens) - 1)
        return padded_tokens

# Load the data into a DataFrame
# data = pd.read_csv('data.csv')

# Define the maximum sequence length
max_len = 256

# Define the dataset
df=None
file1_path = 'dev_test/dev.en'
file2_path = 'dev_test/dev.hi'
with open(file1_path, 'r', errors='ignore') as f1, open(file2_path, 'r', errors='ignore') as f2:
    file1_lines = f1.readlines()
    file2_lines = f2.readlines()

    # Create a dictionary with the data from both files
    data_dict = {'encoder_input': file1_lines, 'decoder_input': file2_lines}

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data_dict)

dataset = CustomDataset(df, max_len)

# Define the vocabulary size
vocab_size = dataset.vocab_size


# Define the training function
def train(model, train_dataloader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for encoder_input_ids, decoder_input_ids in train_dataloader:
            encoder_input_ids = encoder_input_ids.squeeze(1)
            # .to(device)
            decoder_input_ids = decoder_input_ids.squeeze(1)
            # .to(device)

            # Zero out the gradients
            optimizer.zero_grad()

            # Get the model's predictions
            print(encoder_input_ids.size())
            print(decoder_input_ids[:, :-1].size())

            output = model(encoder_input_ids, decoder_input_ids)
            print("output size - ", output[0].size())
            print("decoder input size - ", output[1].size())
            # Compute the loss
            # loss = criterion(output.view(-1, output.shape[-1]), decoder_input_ids[:, 1:].view(-1))
            loss = criterion(output[0], output[1])

            # Backpropagate the gradients
            loss.backward()
            optimizer.step()

            # Add the batch loss to the epoch loss
            epoch_loss += loss.item()
            print("current epoch loss", epoch_loss)

        # Print the epoch loss
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_dataloader)}")

# Load the data into a DataFrame
# data = pd.read_csv('data.csv')

# Define the number of tokens (including the special tokens)
num_tokens = 10000

# Define the model
model = TransformerModel(num_tokens)
# .to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Define the dataset and dataloader
# dataset = CustomDataset(data)
train_dataloader = DataLoader(dataset, batch_size=50)

# Train the model
train(model, train_dataloader, optimizer, criterion, num_epochs=10)
