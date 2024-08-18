import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Define your dataset class
class MyDataset(Dataset):
    def __init__(self, file1, file2):
        self.file1_path = file1
        self.file2_path = file2

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Implement your data loading and preprocessing here
        with open("dev_test/dev.en", "r") as encoder_in:
            encoder_in.readlines().replace("\n")
        

        # Read the contents of the two files into lists
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            file1_lines = f1.readlines()
            file2_lines = f2.readlines()

        # Create a dictionary with the data from both files
        data_dict = {'encoder_in': file1_lines, 'decoder_in': file2_lines}

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(data_dict)
        return processed_data

# Define your transformer model class
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
        return output

# Initialize your model and dataset
vocab_size = 10000
embedding_dim = 512
# num_heads = 8
# num_layers = 6
# from main import Transformer
# model = Transformer(6)
model = TransformerModel(vocab_size)
file1_path = 'dev_test/dev.en'
file2_path = 'dev_test/dev.hi'
dataset = MyDataset(file1_path, file2_path)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train your model
batch_size = 32
num_epochs = 10
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        src, tgt = batch
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("Epoch {} loss: {}".format(epoch+1, epoch_loss / len(train_loader)))

# Evaluate your model on a held-out test set
test_loss = 0
test_loader = DataLoader(test_dataset, batch_size=batch_size)

with torch.no_grad():
    for batch in test_loader:
        src, tgt = batch
        output = model(src, tgt)
        loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
        test_loss += loss.item()
    print("Test loss: {}".format(test_loss / len(test_loader)))
