import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_size = 5000

class BiGramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(vocab_size, vocab_size, bias=False)
        
    def forward(self, x):
        x = F.one_hot(x, num_classes=vocab_size).float()
        out = self.W(x)
        return out