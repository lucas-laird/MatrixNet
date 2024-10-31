import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def comp_y_accs(pred, y):
    rounded = torch.round(pred)
    y0_acc = (rounded[:,0] == y[:,0]).cpu().float().mean()
    y1_acc = (rounded[:,1] == y[:,1]).cpu().float().mean()
    y2_acc = (rounded[:,2] == y[:,2]).cpu().float().mean()
    return y0_acc, y1_acc, y2_acc

def get_nonzero_coords(pred, y, thresh = 0.5):
    nonzero_preds = [torch.nonzero(pred[i]>=thresh).squeeze().tolist() for i in range(pred.size(0))]
    nonzero_truth = [torch.nonzero(y[i]).squeeze().tolist() for i in range(pred.size(0))]
    return (nonzero_preds, nonzero_truth)

def per_y_confusion_matrix(pred, y):
    rounded = torch.round(pred).cpu().detach().numpy()
    y0_cm = confusion_matrix(y[:,0].cpu().numpy(), rounded[:,0])
    y1_cm = confusion_matrix(y[:,1].cpu().numpy(), rounded[:,1])
    y2_cm = confusion_matrix(y[:,2].cpu().numpy(), rounded[:,2])
    return y0_cm, y1_cm, y2_cm

def per_y_avg_dist(pred, y):
    dist_fun = nn.MSELoss()
    loss0 = dist_fun(pred[:,0], y[:,0]).cpu().item()
    loss1 = dist_fun(pred[:,1], y[:,1]).cpu().item()
    loss2 = dist_fun(pred[:,2], y[:,2]).cpu().item()
    return loss0, loss1, loss2


class relationLoss(nn.Module): #relations should be of the form [([r1], [r2]), ([r3], [r4]),...] where r1 = r2 and so on. The ri are lists of generator indices.
    def __init__(self, relations):
        super(relationLoss, self).__init__()
        self.relations = relations
    
    def forward(self, gen_reps, device = 'cuda:0'):
        #matrix_size = torch.sqrt(torch.tensor(mi.size(-1)).to(torch.float)).to(device)
        dists = torch.zeros(len(self.relations)).to(gen_reps.device)
        for j,rel in enumerate(self.relations):
            r1 = rel[0]
            r2 = rel[1]
            A = gen_reps[r1[0], :, :]
            B = gen_reps[r2[0], :, :]
            for i in r1[1:]:
                A = torch.matmul(A, gen_reps[i])
            for i in r2[1:]:
                B = torch.matmul(B, gen_reps[i])
            M = A-B
            dist = torch.linalg.norm(M, dim = (-2,-1))
            dists[j] = dist
            
        loss = torch.mean(dists)
        return loss

def regularize_braid(model, optimizer, device = 'cuda:0'):
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
    