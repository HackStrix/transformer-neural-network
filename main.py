import torch
from torch import nn

import math
print(torch.cuda.is_available())

max_input_length = 256
d_model = 512
batch_size = 50


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
print(encoder_input)

class SelfAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc_q = nn.Linear(512, 256)
        self.fc_k = nn.Linear(512, 256)
        self.fc_v = nn.Linear(512, 256)


    def forward(self, x):
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        return torch.matmul(q,k)
    

x = SelfAttention()

print(x.forward(encoder_input))
