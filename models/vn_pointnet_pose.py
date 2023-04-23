import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature_cross

class get_model(nn.Module):
    def __init__(self, args, normal_channel=False, init_iden=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3

        self.args = args
        self.n_knn = args.n_knn
        
        self.conv_pos = VNLinearLeakyReLU(channel, 64//3, dim=5, negative_slope=0.0)

        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.graph_pool = VNMaxPool(64//3)
            self.pool = VNMaxPool(1024//3)
        elif args.pooling == 'mean':
            self.graph_pool = mean_pool
            self.pool = mean_pool
        
        self.fc3 = VNLinear(256//3, 3)

    def forward(self, x):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)
        x = self.graph_pool(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
#        x = x.view(-1, 3, 3)
#        rot_mat = torch.mean(R, dim=-1)
        
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

#class STNkd(nn.Module):
#    def __init__(self, args, d=64):
#        super(STNkd, self).__init__()
#        self.args = args
#        
#        self.conv1 = VNLinearLeakyReLU(d, 64//3, dim=4, negative_slope=0.0)
#        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
#        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0)
#
#        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0)
#        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0)
#        
#        if args.pooling == 'max':
#            self.pool = VNMaxPool(1024//3)
#        elif args.pooling == 'mean':
#            self.pool = mean_pool
#        
#        self.fc3 = VNLinear(256//3, d)
#        self.d = d
#
#    def forward(self, x):
#        batchsize = x.size()[0]
#        x = self.conv1(x)
#        x = self.conv2(x)
#        x = self.conv3(x)
#        x = self.pool(x)
#
#        x = self.fc1(x)
#        x = self.fc2(x)
#        x = self.fc3(x)
#        
#        return x
#
#
#class PointNetEncoder(nn.Module):
#    def __init__(self, args, global_feat=True, feature_transform=False, channel=3):
#        super(PointNetEncoder, self).__init__()
#        self.args = args
#        self.n_knn = args.n_knn
#        
#        # Modified in channels to be channel (instead of 3) to accomodate for normals
#        self.conv_pos = VNLinearLeakyReLU(channel, 64//3, dim=5, negative_slope=0.0)
#        
#        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
#        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3, dim=4, negative_slope=0.0)
#        
#        self.conv3 = VNLinear(128//3, 1024//3)
#        self.bn3 = VNBatchNorm(1024//3, dim=4)
#        
#        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False, negative_slope=0.0)
#        
#        if args.pooling == 'max':
#            self.pool = VNMaxPool(64//3)
#        elif args.pooling == 'mean':
#            self.pool = mean_pool
#        
#        self.global_feat = global_feat
#        self.feature_transform = feature_transform
#        
#        if self.feature_transform:
#            self.fstn = STNkd(args, d=64//3)
#
#    def forward(self, x):
#        B, D, N = x.size()
#        
#        x = x.unsqueeze(1)
#        feat = get_graph_feature_cross(x, k=self.n_knn)
#        x = self.conv_pos(feat)
#        x = self.pool(x)
#        
#        x = self.conv1(x)
#        
#        if self.feature_transform:
#            x_global = self.fstn(x).unsqueeze(-1).repeat(1,1,1,N)
#            x = torch.cat((x, x_global), 1)
#        
#        pointfeat = x
#        x = self.conv2(x)
#        x = self.bn3(self.conv3(x))
#        
#        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
#        x = torch.cat((x, x_mean), 1)
#        x, trans = self.std_feature(x)
#        x = x.view(B, -1, N)
#        
#        x = torch.max(x, -1, keepdim=False)[0]
#        
#        trans_feat = None
#        if self.global_feat:
#            return x, trans, trans_feat
#        else:
#            x = x.view(-1, 1024, 1).repeat(1, 1, N)
#            return torch.cat([x, pointfeat], 1), trans, trans_feat
