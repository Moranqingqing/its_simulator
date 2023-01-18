import torch
import torch.nn as nn
from model.AGCN import AVWGCN

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num # N
        self.hidden_dim = dim_out # F_O
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim) # F_I+F_O, 2F_O, K, d
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim) # F_I+F_O, F_O, K, d

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim. B*N*F_I
        #state: B, num_nodes, hidden_dim. B*N*H
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1) # [x_t, h_{t-1}], B*N*(F_I+H)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings)) # hat h_t: candidate hidden state
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)