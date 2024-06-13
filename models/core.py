import math
import torch
from models.gnn import MultiGCN
from torch import nn
import torch.nn.functional as F
from models.decoder import DecoderLayer
from torch.nn.utils.weight_norm import weight_norm
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


class LinearAttention(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32, heads=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads

        self.linear_first = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_second = torch.nn.Linear(self.hidden_dim, self.heads)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, masks):
        sentence_att = F.tanh(self.linear_first(x))
        sentence_att = self.linear_second(sentence_att)
        sentence_att = sentence_att.transpose(1, 2)
        minus_inf = -9e15 * torch.ones_like(sentence_att)
        e = torch.where(masks > 0.5, sentence_att, minus_inf)  # (B,heads,seq_len)
        att = self.softmax(e)
        sentence_embed = att @ x
        avg_sentence_embed = torch.sum(sentence_embed, 1) / self.heads

        return avg_sentence_embed



class PMMRNet(torch.nn.Module):
    def __init__(self, args):

        super(PMMRNet, self).__init__()

        self.gnn_dim = args.compound_gnn_dim
        self.dropout = args.dropout
        self.decoder_dim = args.decoder_dim
        self.decoder_heads = args.decoder_heads
        self.compound_text_dim = args.compound_text_dim
        self.compound_structure_dim = args.compound_structure_dim
        self.protein_dim = args.protein_dim
        self.linear_heads = args.linear_heads
        self.linears_hidden_dim = args.linear_hidden_dim
        self.feedforward_dim = args.pf_dim
        self.encoder_heads = args.encoder_heads
        self.encoder_layers = args.encoder_layers
        self.protein_pretrained_dim = args.protein_pretrained_dim
        self.compound_pretrained_dim = args.compound_pretrained_dim
        self.objective = args.objective



        # convolution layers
        self.drug_gcn = MultiGCN(self.compound_structure_dim,  self.gnn_dim)

        self.cross_atten = DecoderLayer(self.decoder_dim, self.decoder_heads, self.dropout)


        self.drug_attn = LinearAttention(self.compound_text_dim, self.linears_hidden_dim, self.linear_heads)
        self.target_attn = LinearAttention(self.protein_dim, self.linears_hidden_dim, self.linear_heads)
        self.inter_attn_one = LinearAttention(self.protein_dim, self.linears_hidden_dim, self.linear_heads)



        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.compound_text_dim, dim_feedforward=self.feedforward_dim, nhead=self.encoder_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.encoder_layers)

        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.protein_dim, dim_feedforward=self.feedforward_dim, nhead=self.encoder_heads)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=self.encoder_layers)


        self.fc1 = nn.Linear(self.compound_structure_dim, self.compound_text_dim)
        self.fc2 = nn.Linear(self.protein_pretrained_dim,self.protein_dim)
        self.fc3 = nn.Linear(self.compound_pretrained_dim,self.compound_text_dim)

        self.drug_ln = nn.LayerNorm(self.compound_text_dim)
        self.target_ln = nn.LayerNorm(self.protein_dim)


        if self.objective == 'regression':
            self.lin = nn.Sequential(
                nn.Linear(self.protein_dim * 3, 1024),
                nn.ReLU(),
                nn.Dropout(self.dropout),

                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),

                nn.Linear(256, 1)

            )
        elif self.objective == 'classification':
            self.lin = nn.Sequential(nn.Linear(self.protein_dim * 3, 512), nn.ReLU(), nn.Dropout(self.dropout),
                                    nn.Linear(512, 2))


    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        batch_size = len(atom_num)
        compound_mask = torch.zeros((batch_size, compound_max_len)).type_as(atom_num)
        protein_mask = torch.zeros((batch_size, protein_max_len)).type_as(atom_num)

        for i in range(batch_size):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        return compound_mask.cuda(), protein_mask.cuda()

    def forward(self, data):

        compound_x = data['COMPOUND_NODE_FEAT'].cuda()
        compound_adj = data['COMPOUND_ADJ'].cuda()
        compound_emb = data['COMPOUND_EMBEDDING'].cuda()

        target_emb = data['PROTEIN_EMBEDDING'].cuda()
        # target_adj = data['PROTEIN_CONTACT_MAP'].cuda()

        compound_smiles_max_len = data['COMPOUND_EMBEDDING'].shape[1]
        compound_node_max_len = data['COMPOUND_NODE_FEAT'].shape[1]

        node_mask, smiles_mask = self.make_masks(
            data["COMPOUND_NODE_NUM"],
            data["COMPOUND_SMILES_LENGTH"],
            compound_node_max_len,
            compound_smiles_max_len,
        )

        #Drug
        compound_struc = self.drug_gcn(compound_x, compound_adj)
        xd_f1 = self.drug_ln(self.fc1(compound_struc))

        # xd_attn_1 = self.drug_attn(xd_f1, compound_mask_1)

        # drug_mask = self.generate_attention_mask(256,4,data['COMPOUND_SMILES_LENGTH'])
        compound_smiles = self.transformer_encoder(self.fc3(compound_emb))
        xd_f2 = self.drug_ln(compound_smiles)
        compound_mask = self.generate_masks(xd_f2, data["COMPOUND_SMILES_LENGTH"], 10)

        xd = self.cross_atten(xd_f2, xd_f1, smiles_mask, node_mask)

        # compound_mask = self.generate_masks(xd, data['COMPOUND_SMILES_LENGTH'], 10)
        xd_attn = self.drug_attn(xd, compound_mask)

        #Protein
        # target_mask = self.generate_attention_mask(256, 4, data["PROTEIN_NODE_NUM"])
        seq_emb = self.transformer_encoder2(self.fc2(target_emb))
        xt = self.target_ln(seq_emb)
        protein_mask = self.generate_masks(xt, data["PROTEIN_NODE_NUM"], 10)
        xt_attn = self.target_attn(xt,protein_mask)

        cat_f = torch.cat([xt, xd], dim=1)
        cat_mask = torch.cat([protein_mask, compound_mask], dim=-1)
        cat_attn = self.inter_attn_one(cat_f, cat_mask)

        #add some dense layers
        out = self.lin(torch.cat([xd_attn,cat_attn,xt_attn],dim=-1))

        return out