import torch
import torch.nn as nn

vocab_size = 5000

class BiGramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.C = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, x):
        out = self.C(x)
        return out