import torch
from torch import nn

import math
print(torch.cuda.is_available())

max_input_length = 256
d_model = 512
batch_size = 50
num_heads = 8
d_k = 128
d_v = int(d_model/num_heads)
d_ff = 2048

input_embeddings = torch.rand(max_input_length, d_model)
print(input_embeddings.size())

def generate_positional_embeddings() -> torch.Tensor:
    positional_embeddings = torch.zeros(max_input_length, d_model)
    for pos, array in enumerate(positional_embeddings):
        for i, _ in enumerate(array):
            if i%2 == 0:
                positional_embeddings[pos, i] = math.sin(pos/(10000**((i)/d_model)))
            else:
                positional_embeddings[pos, i] = math.sin(pos/(10000**((i-1)/d_model)))
    
    # print(positional_embeddings[1, :])
    return positional_embeddings
                
positional_embeddings = generate_positional_embeddings()

encoder_input = input_embeddings + positional_embeddings

print(encoder_input.size())
# print(encoder_input)


class MultiHeadAttention(nn.Module):
    def __init__(self, mask, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heads = [SelfAttention(mask=mask) for _ in range(num_heads)]
        self.fc_o = nn.Linear(num_heads * d_v, d_model)

    def forward(self, x):
        attention_heads = [attention(x) for attention in self.heads]
        multihead = self.fc_o(torch.concat(attention_heads, dim=1))
        return multihead

class SelfAttention(nn.Module):
    def __init__(self, mask, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc_q = nn.Linear(d_model, d_k)
        self.fc_k = nn.Linear(d_model, d_k)
        self.fc_v = nn.Linear(d_model, d_v)
        self.softmax = nn.Softmax(dim=1) ## TODO look into this dimentions again.
        self.mask = mask

    def scaledDotProduct(self, q, k, v, mask:bool= False):
        mul = torch.matmul(q,k.T)
        scale = torch.div(mul, math.sqrt(d_model))
        if mask:
            pass
        else:
            mask = scale
        softmax = self.softmax(mask)
        matmul = torch.matmul(softmax, v)
        return matmul

    def forward(self, x):
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        return self.scaledDotProduct(q, k, v, self.mask)
    

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.non_linear = nn.Sigmoid() # can replace this with relu here, test it out.
        # self.non_linear = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.non_linear(out)
        out = self.fc2(out)
        return out

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.multi_head = MultiHeadAttention(mask=False)
        self.norm = nn.LayerNorm((max_input_length, d_model), elementwise_affine=True)
        self.ffn = FeedforwardNeuralNetModel(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm((max_input_length, d_model), elementwise_affine=True)

    def forward(self, x):
        input = x
        x = self.multi_head(x)
        x = x + input
        x = self.norm(x)
        ffn_input = x
        x = self.ffn(x)
        x = x + ffn_input
        x = self.norm2(x)
        print(x.size())
        return x


attention = Encoder()
x = attention.forward(encoder_input)

def create_look_ahead_mask(size: tuple):
    mask = 1 - torch.triu(torch.ones(size), diagonal=1)
    return mask

def create_padding_mask(seq):
    mask = (seq == 0)
    mask = mask
    # .unsqueeze(1).unsqueeze(2)
    return mask

print(create_look_ahead_mask((256,512)))
print(create_padding_mask(x))
print(x)
print(x.size())
