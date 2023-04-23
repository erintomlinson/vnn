import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

EPS = 1e-10

class get_model(nn.Module):
    def __init__(self, args, normal_channel=False, init_iden=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.init_iden = init_iden

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        if self.init_iden:
            iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
                batchsize, 1)
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden
        x = x.view(-1, 3, 3)
        return x

class get_loss(nn.Module):
    '''
    adapted from https://github.com/orenkatzir/VN-SPD/blob/master/models/networks.py
    '''
    def __init__(self, which_rot_metric='cosine'):
        super().__init__()
        self.identity_rot = torch.eye(3, device='cuda').unsqueeze(0)
        self.which_rot_metric = which_rot_metric
        
        if self.which_rot_metric == 'cosine':
            self.metric = torch.nn.CosineSimilarity(dim=2)
        else:
            self.metric = torch.nn.MSELoss()

    def batched_trace(self, mat):
        return mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

    def cosine_loss(self, R1, R2):
        return torch.mean(1 - self.metric(R1, R2))

    def angular_loss(self, R1, R2):
        M = torch.matmul(R1, R2.transpose(1, 2))
        return torch.mean(torch.acos(torch.clamp((self.batched_trace(M) - 1) / 2., -1 + EPS, 1 - EPS)))

    def forward(self, R1, R2):
        '''
        Input:
            R1, R2 - Bx3x3
        Output:
            loss - torch tensor
        '''
        if self.which_rot_metric == 'cosine':
            return self.cosine_loss(R1, R2)
        elif self.which_rot_metric == 'angular':
            return self.angular_loss(R1, R2)
        elif self.which_rot_metric == 'orthogonal':
            return self.metric(torch.matmul(R1, R2.transpose(1, 2)), self.identity_rot.expand(R1.size(0), 3, 3))
        else:
            return self.metric(R1, R2)


#def feature_transform_reguliarzer(trans):
#    d = trans.size()[1]
#    I = torch.eye(d)[None, :, :]
#    if trans.is_cuda:
#        I = I.cuda()
#    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
#    return loss
