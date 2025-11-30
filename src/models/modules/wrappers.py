import torch.nn as nn
from src.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class Seq2SeqWrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output