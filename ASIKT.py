import torch
from torch import nn
import torch.nn.functional as F
from transformer import TransformerLayer
from mamba2 import Mamba2Config, Mamba2
from run import device


class ASIKT(nn.Module):
    def __init__(self, n_kc, n_q, d_model, dropout, n_blocks, rasch,
                 final_fc_dim=512, n_heads=8, d_ff=2048):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block, default=128
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
        """
        self.n_kc = n_kc
        self.dropout = dropout
        self.n_q = n_q
        self.n_blocks = n_blocks
        embed_l = d_model

        # n_question + 1 (include question_id == 0), d_model
        self.q_embed = nn.Embedding(self.n_q + 1, embed_l)
        self.kc_embed = nn.Embedding(self.n_kc + 1, embed_l)
        self.a_embed = nn.Embedding(2, embed_l)
        self.rasch = rasch

        # rasch embedding for specific dataset
        if self.rasch:
            self.deviates_para = nn.Embedding(self.n_q + 1, 1)
            self.kc_embed_deviates = nn.Embedding(self.n_kc + 1, embed_l)
            self.a_embed_deviates = nn.Embedding(2, embed_l)

        # Architecture of ASIKT. It contains DAKR and KSM blocks
        self.model = Architecture(n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                  d_model=d_model, d_ff=d_ff)

        # Prediction Network
        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        # Slipping
        self.slipping = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        # Guessing
        self.guessing = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_q + 1:
                torch.nn.init.constant_(p, 0.)

    def forward(self, q_data, kc_data, a_data, target, q_dif_mask, qa_dif_mask):
        # Batch First
        q_embed_data = self.q_embed(q_data)
        kc_embed_data = self.kc_embed(kc_data)  # BS, seqlen, d_model
        a_embed_data = self.a_embed(a_data)  # 0 0 0 1...

        # BS, seqlen, d_model
        kca_embed_data = kc_embed_data + a_embed_data
        qa_embed_data = q_embed_data + a_embed_data

        # rasch embedding
        if self.rasch:
            kc_embed_deviates_data = self.kc_embed_deviates(kc_data)
            a_embed_deviates_data = self.a_embed_deviates(a_data)
            q_deviates_para = self.deviates_para(q_data)

            q_embed_data = kc_embed_data + q_deviates_para * kc_embed_deviates_data
            qa_embed_data = kca_embed_data + q_deviates_para * (a_embed_deviates_data + kc_embed_deviates_data)

        # output shape: BS, seqlen, d_model * 4
        d_output = self.model(q_embed_data, kc_embed_data, qa_embed_data, kca_embed_data,
                              q_dif_mask, qa_dif_mask)

        # Residual connection to Prediction Network
        concat_kc = torch.cat([d_output[0], kc_embed_data], dim=-1)

        state_output = self.out(concat_kc)
        slipping = self.slipping(concat_kc)
        guessing = self.guessing(concat_kc)

        # consider Slipping and Guessing
        output = (1 - state_output) * guessing + state_output * (1 - slipping)

        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = (output.reshape(-1))  # logit
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        return output.sum(), m(preds), mask.sum()


class Architecture(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout):
        super().__init__()
        """
            n_block : number of stacked blocks
            d_model : dimension of model input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.linear = nn.Linear(d_model * 4, d_model)

        ## DAKR blocks
        # q encoder
        self.blocks_1 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads)
            for _ in range(n_blocks)])
        # qa encoder
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads)
            for _ in range(n_blocks)])
        # kc encoder
        self.blocks_3 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads)
            for _ in range(n_blocks)])
        # kca encoder
        self.blocks_4 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads)
            for _ in range(n_blocks)])

        ## KSM blocks
        # knowledge retriever
        self.blocks_5 = Mamba2(args=Mamba2Config(d_model=d_model * 4), device=device)
        self.blocks_6 = Mamba2(args=Mamba2Config(d_model=d_model), device=device)

    def forward(self, q_embed_data, kc_embed_data, qa_embed_data, kca_embed_data,
                q_dif_mask, qa_dif_mask):
        # target shape  bs, seqlen
        batch_size, seqlen, d_model = kca_embed_data.size(0), kca_embed_data.size(1), kca_embed_data.size(2)

        q = q_embed_data
        kc = kc_embed_data
        qa = qa_embed_data
        kca = kca_embed_data

        ## DAKR blocks
        # encode q # bs, sql, d_model
        for block in self.blocks_1:
            q = block(mask=1, query=q, key=q, values=q, dif_mask=q_dif_mask)
        # # encode qa # bs, sql, d_model
        for block in self.blocks_2:
            qa = block(mask=1, query=qa, key=qa, values=qa, dif_mask=qa_dif_mask)
        # encode kc # bs, sql, d_model
        for block in self.blocks_3:
            kc = block(mask=1, query=kc, key=kc, values=kc, dif_mask=q_dif_mask)
        # encode kca # bs, sql, d_model
        for block in self.blocks_4:
            kca = block(mask=1, query=kca, key=kca, values=kca, dif_mask=qa_dif_mask)

        # shift to get the input: qa(t) + kca(t) + q(t+1) + kc(t+1)
        q_shifted = q[:, 1:, :]
        kc_shifted = kc[:, 1:, :]
        zero_vector_q = torch.zeros((batch_size, 1, d_model), device=device)
        zero_vector_kc = torch.zeros((batch_size, 1, d_model), device=device)
        q_padded = torch.cat((q_shifted, zero_vector_q), dim=1)
        kc_padded = torch.cat((kc_shifted, zero_vector_kc), dim=1)

        xa = torch.cat([qa, kca, q_padded, kc_padded], dim=-1)

        # KSM blocks
        # bs, sql, d_model * 4
        h = self.blocks_5(xa)
        h = F.sigmoid(self.linear(h[0]))
        y = self.blocks_6(h)

        return y
