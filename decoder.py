import torch
from torch import nn
from layers import MultiHeadAttention, FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads=4, fc_dim=512, dropout=0.1, device='cpu'):
        super(DecoderLayer, self).__init__()

        # self-attention
        self.attn1 = MultiHeadAttention(model_dim, model_dim, model_dim, model_dim, num_heads, device)
        self.norm1 = nn.LayerNorm(model_dim)
        self.drop1 = nn.Dropout(dropout)

        # encoder - decoder attention
        self.attn2 = MultiHeadAttention(model_dim, model_dim, model_dim, model_dim, num_heads, device)
        self.norm2 = nn.LayerNorm(model_dim)
        self.drop2 = nn.Dropout(dropout)

        # fully connected
        self.fc = FeedForward(model_dim, fc_dim, device)
        self.norm3 = nn.LayerNorm(model_dim)
        self.drop3 = nn.Dropout(dropout)


    def forward(self, memory, target, src_mask, tgt_mask):
        """
        :param memory: Of shape [batch_size, T1, dim]
        :param target: Of shape [batch_size, T2, dim]
        :param src_mask: Of shape [batch_size, T1, T1]
        :param tgt_mask: Of shape [batch_size, T2, T2]
        :return:
        """
        batch_size, T2, dim = target.shape
        tgt_mask = torch.tril(torch.ones(batch_size,T2,T2)).int() & tgt_mask.int()[:,None,:]
        tgt_mask = tgt_mask.int()
        target = self.norm1(
            target + self.drop1(
                self.attn1(target, target, target, tgt_mask)
            )
        )

        target = self.norm2(
            target + self.drop2(
                self.attn2(target, memory, memory, src_mask[:,None,:])
            )
        )

        target = self.norm3(
            target + self.drop3(
                self.fc(target)
            )
        )

        return target

class Decoder(nn.Module):
    def __init__(self,  model_dim, num_layers, num_heads=4, fc_dim=512, dropout=0.1, device='cpu'):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(model_dim, num_heads, fc_dim, dropout, device)
            for _ in range(num_layers)
        ])

    def forward(self, memory, target, src_mask, tgt_mask):
        for layer in self.layers:
            target = layer(memory, target, src_mask, tgt_mask)
        return target


def decoder_test():

    batch_size = 1
    T2 = 7
    T1 = 6
    dim = 4

    tgt = torch.randint(0, 5, size=[batch_size, T2])
    tgt_mask = (tgt != 0).int()

    src = torch.randint(0, 5, size=[batch_size, T1])
    src_mask = (src != 0).int()

    tgt_embedding = torch.randn(batch_size, T2, dim)
    src_embedding = torch.randn(batch_size, T1, dim)

    # model = DecoderLayer(dim, 1, fc_dim=16)
    model = Decoder(dim, 2, fc_dim=16)
    y = model(src_embedding, tgt_embedding, src_mask, tgt_mask)
    print(y.shape)



# decoder_test()