import torch
import torch.nn as nn
from torch import matrix_exp
import math

class MatrixNet(nn.Module):
    #Recurrent/Residual model that takes in a braid word and sequentially learns an invertible matrix representation of each element in the word. 
    #The matrices are right multiplied in sequence to get a final output matrix representation of the braid word. 
    def __init__(self, num_generators, hidden_rep_size, matrix_size, hidden_class_size, output_size, matrix_channels = 1, signed = False, matrix_block_type = None):
        super(MatrixNet, self).__init__()
        if signed:
            self.input_size = num_generators
        else:
            self.input_size = num_generators*2
        self.identity_ele = torch.zeros(self.input_size)
        self.matrix_channels = matrix_channels
        self.hidden_rep_size = hidden_rep_size
        self.matrix_size = matrix_size
        self.total_mat_size = matrix_channels * (matrix_size**2)
        self.hidden_class_size = hidden_class_size
        self.output_size = output_size
        self.signed = signed
        self.matrix_block_layers = nn.ModuleList()
        if matrix_block_type == '2-layer':
            self.matrix_block_layers.append(nn.Linear(self.input_size, hidden_rep_size, bias = False))
            self.matrix_block_layers.append(nn.Linear(hidden_rep_size, self.total_mat_size, bias = False))
        elif matrix_block_type == 'tanh':
            self.matrix_block_layers.append(nn.Linear(self.input_size, hidden_rep_size, bias = False))
            self.matrix_block_layers.append(nn.Linear(hidden_rep_size, self.total_mat_size, bias = False))
            self.matrix_block_layers.append(nn.Tanh())
        elif matrix_block_type == '1-layer': 
            self.matrix_block_layers.append(nn.Linear(self.input_size, self.total_mat_size, bias = False))

        self.nonlin = nn.ReLU()
        self.l1_class = nn.Linear(self.total_mat_size, hidden_class_size)
        self.l2_class = nn.Linear(hidden_class_size, hidden_class_size)
        self.out = nn.Linear(hidden_class_size, output_size)
        
        
    def matrix_block(self, x):
        #Res/Recurrent block that computes matrix
        #Find all non-identity braid elements
        for (i, layer) in enumerate(self.matrix_block_layers):
            x = layer(x)
        x = x.reshape((-1, self.matrix_channels, self.matrix_size, self.matrix_size))
        x = matrix_exp(x)
        return x
    
    def forward(self, x, word_length = -1):
        if word_length == -1:
            braid_length = x.size(1)
        else:
            word_length = braid_length
        braid = x[:,0,:]
        mat_rep = self.matrix_block(braid)
        for i in range(1,braid_length):
            braid = x[:,i,:]
            temp = self.matrix_block(braid)
            mat_rep = torch.matmul(mat_rep, temp)
        flattened_rep = torch.reshape(mat_rep, (-1, self.total_mat_size))
        output = self.nonlin(self.l1_class(flattened_rep))
        output = self.nonlin(self.l2_class(output))
        output = self.out(output)
        return output


            




class BraidMLP(nn.Module):
    #Simple MLP that takes in flattened onehot encodings.
    def __init__(self, num_generators, max_length, hidden_size, output_size, nlayers, signed = False):
        super(BraidMLP, self).__init__()
        if signed:
            self.input_size = num_generators*max_length
        else:
            self.input_size = num_generators*2*max_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.signed = signed
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, hidden_size))
        for i in range(nlayers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.out = nn.Linear(hidden_size, output_size)
        self.nonlin = nn.ReLU()
    def forward(self,x):
        for (i,l) in enumerate(self.layers):
            x = l(x)
            x = self.nonlin(x)
        output = self.out(x)
        return output       


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BraidTransformer(nn.Module):

    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, output_dim = 3, dropout = 0.5):
        super(BraidTransformer, self).__init__()
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len = 10)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model, padding_idx = 0)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, output_dim)

    
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src.permute(1,0,2)
        src = self.pos_encoder(src)
        src = src.permute(1,0,2)
        output = self.transformer_encoder(src)
        output = output.mean(dim = 1)
        output = self.linear(output)
        return output

        


class BraidLSTM(nn.Module):
    def __init__(self, input_dim, ntoken, hidden_dim, num_layers, hidden_class, output_dim):
        super(BraidLSTM, self).__init__()
        self.embedding = nn.Embedding(ntoken, input_dim, padding_idx = 0)
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.classifier = nn.ModuleList([nn.Linear(hidden_dim, hidden_class),
                                        nn.ReLU(),
                                        nn.Linear(hidden_class, output_dim)])
    def forward(self, x):
        x = self.embedding(x)
        hiddens, hidden_final = self.lstm_layer(x)
        out = hidden_final[0]
        out = out.mean(dim = 0) #mean over layer features
        for (i,l) in enumerate(self.classifier):
            out = l(out)
        return out
            
            
