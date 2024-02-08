import torch
import torch.nn as nn
from torch import matrix_exp

class BraidResNet(nn.Module):
    #Recurrent/Residual model that takes in a braid word and sequentially learns an invertible matrix representation of each element in the word. 
    #The matrices are right multiplied in sequence to get a final output matrix representation of the braid word. 
    def __init__(self, num_generators, hidden_rep_size, matrix_size, hidden_class_size, output_size, signed = False):
        super(BraidResNet, self).__init__()
        if signed:
            self.input_size = num_generators
        else:
            self.input_size = num_generators*2
        self.identity_ele = torch.zeros(self.input_size)
        self.hidden_rep_size = hidden_rep_size
        self.matrix_size = matrix_size
        self.hidden_class_size = hidden_class_size
        self.output_size = output_size
        self.signed = signed
        self.l1 = nn.Linear(self.input_size, hidden_rep_size)
        self.l2 = nn.Linear(hidden_rep_size, hidden_rep_size)
        self.l3 = nn.Linear(hidden_rep_size, matrix_size**2)
        self.nonlin = nn.ReLU()
        self.l1_class = nn.Linear(matrix_size**2, hidden_class_size)
        self.l2_class = nn.Linear(hidden_class_size, hidden_class_size)
        self.out = nn.Linear(hidden_class_size, output_size)
        
    def res_block(self, x):
        #Res/Recurrent block that computes matrix
        #Find all non-identity braid elements
        nonzero_elements = torch.nonzero(torch.all(x == self.identity_ele, dim = 1))
        mat = self.nonlin(self.l1(x))
        mat = self.nonlin(self.l2(mat))
        mat = self.l3(mat)
        mat = mat.reshape((-1,self.matrix_size, self.matrix_size))
        mat = matrix_exp(x)
        #identity_matrix = torch.eye(self.matrix_size, self.matrix_size)
        #mat[nonzero_elements, :, :] = 
        return mat
    
    def forward(self, x, word_length = -1):
        if word_length == -1:
            braid_length = x.size(1)
        else:
            word_length = braid_length
        braid = x[:,0,:]
        mat_rep = self.res_block(braid)
        for i in range(1,braid_length):
            braid = x[:,i,:]
            temp = self.res_block(braid)
            mat_rep = torch.matmul(mat_rep, temp)
        flattened_rep = torch.reshape(mat_rep, (-1, self.matrix_size**2))
        output = self.nonlin(self.l1_class(flattened_rep))
        output = self.nonlin(self.l2_class(output))
        output = self.out(output)
        return output


class BraidMLP(nn.Module):
    #Simple MLP that takes in flattened onehot encodings.
    def __init__(self, num_generators, max_length, hidden_size, output_size, signed = False):
        super(BraidMLP, self).__init__()
        if signed:
            self.input_size = num_generators*max_length
        else:
            self.input_size = num_generators*2*max_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.signed = signed
        self.l1 = nn.Linear(self.input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.nonlin = nn.ReLU()
    def forward(self,x):
        x = self.l1(x)
        x = self.nonlin(x)
        x = self.l2(x)
        x = self.nonlin(x)
        x = self.l3(x)
        x = self.nonlin(x)
        x = self.nonlin(self.l4(x))
        output = self.out(x)
        return output         

