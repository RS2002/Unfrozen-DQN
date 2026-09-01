import torch.nn as nn
import torch
import math

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim=2, hidden_dims=[256,256,256], dropout_rate=0.3):
        super().__init__()
        self.model=nn.ModuleList()
        self.model.append(nn.Linear(state_dim,hidden_dims[0]))
        for i in range(1,len(hidden_dims)):
            self.model.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.model.append(nn.Linear(hidden_dims[-1],action_dim))
        self.relu=nn.ReLU()
        self.drop=nn.Dropout(dropout_rate)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=x.to(torch.float32)
        for layer in self.model[:-1]:
            x=layer(x)
            x=self.relu(x)
            x=self.drop(x)
        x=self.model[-1](x)
        #x=self.sigmoid(x)
        return x

    '''
    Description: 
        freeze the first <prob> parameters in each layer
    Input:
        prob: the proportion of parameters to freeze
    Implement:
        clear the grad of parameters to be frozen
    '''
    def freeze(self,prob):
        if prob<=0:
            return
        for layer in self.model:
            params=layer.parameters()
            for p in params:
                num=int(p.shape[0]*prob)
                p.grad[0:num]=0

    def open_grid(self):
        self.requires_grad_(True)

    def close_grid(self):
        self.requires_grad_(False)


class DQN_Attention(nn.Module):
    def __init__(self, state_dim, action_dim=2, hidden_dims=[256], num_heads=2, emb_dims=4, dropout_rate=0.3, device=torch.device("cpu")):
        super().__init__()
        self.state_dim=state_dim
        self.emb_dims=emb_dims
        self.emb=nn.Embedding(3,emb_dims)
        self.attention=nn.MultiheadAttention(embed_dim=emb_dims,num_heads=num_heads,dropout=dropout_rate)
        self.Q = nn.Linear(emb_dims, emb_dims)
        self.K = nn.Linear(emb_dims, emb_dims)
        self.V = nn.Linear(emb_dims, emb_dims)
        self.model=nn.ModuleList()
        self.model.append(nn.Linear(emb_dims*state_dim,hidden_dims[0]))
        for i in range(1,len(hidden_dims)):
            self.model.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.model.append(nn.Linear(hidden_dims[-1],action_dim))
        self.relu=nn.ReLU()
        self.drop=nn.Dropout(dropout_rate)
        self.sigmoid=nn.Sigmoid()
        self.pos_emb=self.positional_encoding(state_dim//2,emb_dims)
        self.pos_emb=torch.cat([self.pos_emb,self.pos_emb],dim=-2).to(device)

    def forward(self,x):
        x[x==-1]=torch.max(x)+1
        x = x.to(torch.long)
        if len(x.shape)==1:
            x=x.view(1,-1)
        x=self.emb(x)
        y=x+self.pos_emb
        y,_=self.attention(self.Q(y),self.K(y),self.V(y))
        #y, _ = self.attention(y,y,y)
        y+=x
        x=x.view(-1,self.state_dim*self.emb_dims)
        for layer in self.model[:-1]:
            x=layer(x)
            x=self.relu(x)
            x=self.drop(x)
        x=self.model[-1](x)
        #x=self.sigmoid(x)
        return x.squeeze()

    # the position embedding mechanism in Transformer
    def positional_encoding(self,seq_len, d_model):
        pos_enc = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    '''
    Description: 
        freeze the first <prob> parameters in each layer
    Input:
        prob: the proportion of parameters to freeze
    Implement:
        clear the grad of parameters to be frozen
    '''
    def freeze(self,prob):
        if prob<=0:
            return
        for layer in self.model:
            params=layer.parameters()
            for p in params:
                num=int(p.shape[0]*prob)
                p.grad[0:num]=0

    def open_grid(self):
        self.requires_grad_(True)

    def close_grid(self):
        self.requires_grad_(False)
