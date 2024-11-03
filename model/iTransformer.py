import torch
import torch.nn as nn
from einops import rearrange
from layers.Embed import DataEmbedding_inverted


class Dual(nn.Module):
    def __init__(
        self,
        device="cuda:7",
        channel=768,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=768,
        e_layer=1,
        d_ff=32,
        head=8
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n = dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_ff = d_ff
        self.head = head

        self.enc_embedding = DataEmbedding_inverted(self.seq_len, d_model=self.channel, dropout=self.dropout_n).to(self.device)

        # Time Series Encoder
        self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
                                                           norm_first = True,dropout = self.dropout_n).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Projection
        self.projector = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, x, x_mark):
        ts_data = x.float() # B L N
        # print(ts_data.shape)
        x_mark = x_mark.float()

        # Norm
        means = ts_data.mean(1, keepdim=True).detach()
        ts_data = ts_data - means
        stdev = torch.sqrt(torch.var(ts_data, dim=1, keepdim=True, unbiased=False) + 1e-5)
        ts_data /= stdev
        
        _, _, N = ts_data.shape

        # Emb
        ts_emb_out = self.enc_embedding(ts_data, x_mark)

        # Prompt Encoder
        ts_enc_out = self.ts_encoder(ts_emb_out)

        # Projection
        dec_out = self.projector(ts_enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        # Denorm
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out