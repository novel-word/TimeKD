import torch.nn as nn
from einops import rearrange


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
        self.head = head

        # self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.token_to_feature = nn.Linear(self.d_llm, self.channel).to(self.device)

        # Time Series Encoder
        # self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
        #                                                    norm_first = True,dropout = self.dropout_n).to(self.device)
        # self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Prompt Encoder
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
                                                               norm_first = True,dropout = self.dropout_n).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Projection
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, embeddings):
        embeddings = embeddings.float()
        # print(embeddings.shape)
        embeddings = embeddings.squeeze(-1) # B N E
        embeddings = self.token_to_feature(embeddings) # B N C
        embeddings_out = self.prompt_encoder(embeddings)
        rec_out = self.c_to_length(embeddings_out)
        rec_out = rec_out.permute(0, 2, 1)

        return rec_out