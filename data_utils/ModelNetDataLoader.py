import numpy as np
import warnings
import os
from torch.utils.data import Dataset
import torch
from pytorch3d.transforms import RotateAxisAngle
warnings.filterwarnings('ignore')



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000, subset='modelnet40', class_choice='airplane', align=True):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, f'{subset}_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.class_names = {v: k for k, v in self.classes.items()}
        self.normal_channel = normal_channel
        self.class_choice = class_choice
        self.align = align

        if self.align:
            shape_ids = {}
            shape_ids['train'] = [line.rstrip().split() for line in open(os.path.join(self.root, f'{subset}_train_pose.txt')) if self.class_choice in line]
            shape_ids['test'] = [line.rstrip().split() for line in open(os.path.join(self.root, f'{subset}_test_pose.txt')) if self.class_choice in line]

            assert (split == 'train' or split == 'test')
            shape_poses = [x[1] for x in shape_ids[split]]
            shape_ids[split] = [x[0] for x in shape_ids[split]]
            shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
            # list of (shape_name, shape_txt_file_path, pose) tuple
            self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt', shape_poses[i]) for i
                             in range(len(shape_ids[split])) if shape_poses[i] != 'x']
            print('The size of aligned %s %s data is %d'%(self.class_choice, split, len(self.datapath)))

            self.trot = {
                'u': RotateAxisAngle(angle=0, axis='Y', degrees=True), # do nothing, this is the reference pose
                'l': RotateAxisAngle(angle=90, axis='Y', degrees=True),
                'd': RotateAxisAngle(angle=180, axis='Y', degrees=True),
                'r': RotateAxisAngle(angle=-90, axis='Y', degrees=True)}

        else:
            shape_ids = {}
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, f'{subset}_train.txt')) if self.class_choice in line]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, f'{subset}_test.txt')) if self.class_choice in line]

            assert (split == 'train' or split == 'test')
            shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
            # list of (shape_name, shape_txt_file_path) tuple
            self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                             in range(len(shape_ids[split]))]
            print('The size of %s %s data is %d'%(self.class_choice, split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if self.align:
                point_set = torch.tensor(point_set)
                point_set[:, 0:3] = self.trot[fn[2]].transform_points(point_set[:, 0:3])
                point_set[:, 3:6] = self.trot[fn[2]].transform_points(point_set[:, 3:6])
                point_set = point_set.numpy()

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)
