import torch
import torch.nn as nn

class relationLoss(nn.Module): 
    #relations should be of the form [([r1], [r2]), ([r3], [r4]),...] where r1 = r2 and so on. The ri are lists of generator indices.
    # A None in relation tuple corresponds to the identity.
    def __init__(self, relations):
        super(relationLoss, self).__init__()
        self.relations = relations
    
    def forward(self, gen_reps, device = 'cuda:0'):
        #matrix_size = torch.sqrt(torch.tensor(mi.size(-1)).to(torch.float)).to(device)
        dists = torch.zeros(len(self.relations)).to(gen_reps.device)
        for j,rel in enumerate(self.relations):
            r1 = rel[0]
            r2 = rel[1]
            if r1 is not None:
                A = gen_reps[r1[0], :, :]
                for i in r1[1:]:
                    A = torch.matmul(A, gen_reps[i])
            else:
                A = torch.eye(gen_reps.shape(-2,-1)).to(gen_reps.device)
            if r2 is not None:
                B = gen_reps[r2[0], :, :]
                for i in r2[1:]:
                    B = torch.matmul(B, gen_reps[i])
            else:
                B = torch.eye(gen_reps.shape(-2,-1)).to(gen_reps.device)
            M = A-B
            dist = torch.linalg.norm(M, dim = (-2,-1))
            dists[j] = dist
            
        loss = torch.mean(dists)
        return loss

def regularize_braid(model, device = 'cuda:0'):
    loss_fn = relationLoss([([0,1,0],[1,0,1]), ([2,3,2], [3,2,3])])
    m = torch.zeros(4,1,2)
    m[0,0,0] = 1 #sig1
    m[1,0,1] = 1 #sig2
    m[2,0,0] = -1 #sig1_inv
    m[3,0,1] = -1 #sig2_inv
    m = m.to(device)
    
    gen_reps = model.matrix_block(m)
    
    loss = loss_fn(gen_reps)
    
    return loss

def regularize_symmetric(model, n, device = 'cuda:0'):
    relations = []
    for i in range(n-1):
        # i^2 = 1 self inverse relation
        relations.append(([i,i], None))
    for i in range(n-2):
        #braid relation
        relations.append(([i,i+1,i], [i+1, i, i+1]))
        for j in range(i+2,n-1):
            # non-adjacent transpositions commute relation
            relations.append = (([i,j], [j,i]))
    m = torch.zeros(n-1, 1, n-1)
    for i in range(n-1):
        m[i,0,i] = 1
    m.to(device)
    
    gen_reps = model.matrix_block(m)
    
    loss_fn = relationLoss(relations)
    loss = loss_fn(gen_reps)
    
    return loss