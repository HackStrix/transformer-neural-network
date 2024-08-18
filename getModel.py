import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.embedding_src = nn.Embedding(512, src_vocab_size)
        self.embedding_tgt = nn.Embedding(512, trg_vocab_size)
        from model import Transformer
        self.transformer = Transformer(Nx=6)        
    def forward(self, src, tgt):
        src_embed = self.embedding_src(src)
        tgt_embed = self.embedding_tgt(tgt)
        output = self.transformer(src_embed, tgt_embed)
        output = self.embedding_tgt(output)
        probability = torch.softmax(output, dim=-1)

        # output = self.fc(output)
        return probability


def get_model(opt, src_vocab, trg_vocab):
    
    # assert opt.d_model % opt.heads == 0
    # assert opt.dropout < 1

    model = TransformerModel(src_vocab, trg_vocab)
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.device == 0:
        model = model.cuda()
    
    return model