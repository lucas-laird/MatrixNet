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
    