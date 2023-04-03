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
output_vocab_size = 10000

input_embeddings = torch.rand(max_input_length, d_model)
input_embeddings = input_embeddings.repeat(batch_size, 1, 1)
input_embeddings = torch.rand(batch_size, max_input_length, d_model)

print(input_embeddings.size())

def generate_positional_embeddings() -> torch.Tensor:
    positional_embeddings = torch.zeros(batch_size, max_input_length, d_model)
    for pos, array in enumerate(positional_embeddings[0, :, :]):
        for i, _ in enumerate(array):
            # print(pos, array, i, l)
            if i%2 == 0:
                positional_embeddings[0, pos, i] = math.sin(pos/(10000**((i)/d_model)))
            else:
                positional_embeddings[0, pos, i] = math.sin(pos/(10000**((i-1)/d_model)))

    # print(positional_embeddings[1, :])
    return positional_embeddings

def generate_batch_positional_embeddings() -> torch.Tensor:
    positional_embeddings = generate_positional_embeddings()
    for batch in range(batch_size):
        positional_embeddings[batch, :, :] = positional_embeddings[0, :,:]
    return positional_embeddings
                
positional_embeddings = generate_batch_positional_embeddings()
print(positional_embeddings.size())
# exit()
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
        multihead = self.fc_o(torch.concat(attention_heads, dim=-1))
        return multihead
    
class CrossMultiHeadAttention(nn.Module):
    def __init__(self, mask, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heads = [SelfAttention(mask=mask) for _ in range(num_heads)]
        self.fc_o = nn.Linear(num_heads * d_v, d_model)

    def forward(self, x, y):
        attention_heads = [attention(x, y) for attention in self.heads]
        multihead = self.fc_o(torch.concat(attention_heads, dim=-1))
        return multihead

class SelfAttention(nn.Module):
    def __init__(self, mask, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc_q = nn.Linear(d_model, d_k)
        self.fc_k = nn.Linear(d_model, d_k)
        self.fc_v = nn.Linear(d_model, d_v)
        self.softmax = nn.Softmax(dim=-1) ## TODO look into this dimentions again.
        self.mask = mask

    def scaledDotProduct(self, q, k, v, mask:bool= False):
        k_transpose = torch.transpose(k, 1, 2)
        mul = torch.matmul(q,k_transpose)
        scale = torch.div(mul, math.sqrt(d_model))
        if mask:
            masked = scale + create_look_ahead_mask(scale.size())
        else:
            masked = scale
        softmax = self.softmax(masked)
        matmul = torch.matmul(softmax, v)
        return matmul

    def forward(self, x, y=None):
        q = self.fc_q(x)
        if y is None:
            k = self.fc_k(x)
            v = self.fc_v(x)
        else:
            k = self.fc_k(y)
            v = self.fc_v(y)

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
        self.norm = nn.LayerNorm((d_model, ), elementwise_affine=True)
        self.ffn = FeedforwardNeuralNetModel(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm((d_model, ), elementwise_affine=True)

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




class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.masked_multi_head = MultiHeadAttention(mask=True)
        self.norm = nn.LayerNorm((d_model, ), elementwise_affine=True)
        self.cross_head = CrossMultiHeadAttention(mask=False)
        self.norm2 = nn.LayerNorm((d_model, ), elementwise_affine=True)
        self.ffn = FeedforwardNeuralNetModel(d_model, d_ff, d_model)
        self.norm3 = nn.LayerNorm((d_model, ), elementwise_affine=True)
    

    def forward(self, output_embeddings, encoder_output):
        x = self.masked_multi_head(output_embeddings)
        x = x + output_embeddings
        x = self.norm(x)
        masked_attention = x
        x = self.cross_head(x, encoder_output)
        x = x + masked_attention
        x = self.norm2(x)
        cross_attention = x
        x = self.ffn(x)
        x = x + cross_attention
        x = self.norm3(x)
        print(x.size())
        return x



def create_look_ahead_mask(size: tuple):
    mask = 1 - torch.triu(torch.ones(size), diagonal=1)
    mask[mask == 0] = float('-inf')
    mask[mask == 1] = 0
    return mask

# TODO to create a padding mask and understand where in the architecuture it should be implemented 

# def create_padding_mask(seq):
#     mask = (seq == 0)
#     mask = mask
#     # .unsqueeze(1).unsqueeze(2)
#     return mask


class Transformer(nn.Module):
    def __init__(self, Nx, output_vocab_size  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoders = [Encoder() for _ in range(Nx)]
        self.decoders = [Decoder() for _ in range(Nx)]
        self.fc_o = nn.Linear(d_model, output_vocab_size)

    def forward(self, x, y):
        encoder_out = x
        for encoder in self.encoders:
            encoder_out = encoder(encoder_out)
        
        decoder_out = y
        for decoder in self.decoders:
            decoder_out = decoder(decoder_out, encoder_out)

        # TODO map this to the vocab size using linear layer and softmax
        output = self.fc_o(decoder_out)
        probability = torch.softmax(output, dim=-1)
        return probability



# encoder = Encoder()
# decoder = Decoder()
# encoder_out = encoder.forward(encoder_input)
# print(encoder_out[0,:,:])
# print(encoder_out[1,:,:])
# decoder_out = decoder.forward(encoder_input, encoder_out)

trans = Transformer(6)
print(trans(encoder_input, encoder_input))
# print(decoder_out)


# print(create_look_ahead_mask((256,256)))
# print(create_padding_mask(x))
# print(x)
# print(x.size())
