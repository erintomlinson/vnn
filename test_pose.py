"""
Author: Congyue Deng
Contact: congyue@stanford.edu
Date: April 2021
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import provider
import importlib
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size in training [default: 32]')
    parser.add_argument('--model', default='vn_dgcnn_pose', help='Model name [default: vn_dgcnn_pose]',
                        choices = ['pointnet_pose', 'vn_pointnet_pose', 'dgcnn_pose', 'vn_dgcnn_pose'])
    parser.add_argument('--gpu', type=str, default='0', help='Specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='vn_dgcnn/aligned', help='Experiment root [default: vn_dgcnn/aligned]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--rot', type=str, default='aligned', help='Rotation augmentation to input data [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--pooling', type=str, default='mean', help='VNN only: pooling method [default: mean]',
                        choices=['mean', 'max'])
    parser.add_argument('--n_knn', default=20, type=int, help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    parser.add_argument('--subset', default='modelnet40', type=str, help='Subset to use for training [modelnet10, modelnet40 (default)]')
    parser.add_argument('--class_choice', type=str, default='airplane', help='class choice [default %(default)s]',
                        choices=['airplane'])
    parser.add_argument('--init_iden', action='store_true', help='initialize output to identity')
    parser.add_argument('--acc_threshold', type=float, default=10.0, help='axis-angle distance considered correct [%(default)f]')
#    parser.add_argument('--which_rot_metric', type=str, default='cosine', help='loss selection for rotation loss')
#    parser.add_argument('--which_ortho_metric', type=str, default='MSE', help='loss selection for orthogonal loss')
#    parser.add_argument('--partial_prob_test', nargs='+', default=[0.0], type=float, help='Probability of single-view point cloud conversion for testing [default: 0]')
    parser.add_argument('--num_partials', type=int, default=2, help='Compute test accuracy this many times and take the average [default: 5]')
    parser.add_argument('--num_rots', type=int, default=3, help='Compute test accuracy this many times and take the average [default: 5]')
    parser.add_argument('--renormalize', action='store_true', help='recenter and renormalize after partialization')
    parser.add_argument('--random_shift', action='store_true', help='apply random shifts to point cloud during training')
    return parser.parse_args()

def test(model, loader, num_class=40, num_rots=3, num_partials=2):
    mean_correct = []
    mean_axis_angle_distances = dict(complete=[], partial=[])
    for _ in range(num_rots):

        axis_angle_distances = dict(complete=[], partial=[])
        for j, data in tqdm(enumerate(loader), total=len(loader)):
            points, _ = data
        
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(points.shape[0]))
            if args.normal:
                points[:,:,0:3] = trot.transform_points(points[:,:,0:3])
                points[:,:,3:6] = trot.transform_normals(points[:,:,3:6])
            else:
                points = trot.transform_points(points)
            target_rot = trot.get_matrix()[:, :3, :3]

            points = points.transpose(2, 1)
            points, target_rot = points.cuda(), target_rot.cuda()
            pred_rot = model(points)
            pred_rot = provider.to_rotation_mat(pred_rot.detach())
            batch_axis_angle_distances = provider.axis_angle_distance(pred_rot, target_rot)
            axis_angle_distances['complete'].append(batch_axis_angle_distances)

            partial_batch_axis_angle_distances = torch.zeros_like(batch_axis_angle_distances)
            for _ in range(num_partials): 
                points_ = points.transpose(2, 1).clone()
                points_ = points_.cpu().numpy()
                partial_points, _ = provider.partialize_point_cloud(points_, prob=1.0, renormalize=args.renormalize)
                partial_points = torch.tensor(partial_points)

                partial_points = partial_points.transpose(2, 1)
                partial_points = partial_points.cuda()
                partial_pred_rot = model(partial_points)
                partial_pred_rot = provider.to_rotation_mat(partial_pred_rot.detach())
                partial_batch_axis_angle_distances += provider.axis_angle_distance(partial_pred_rot, target_rot)
            axis_angle_distances['partial'].append(partial_batch_axis_angle_distances / num_partials)
        
        for k in mean_axis_angle_distances.keys():
            mean_axis_angle_distances[k].append(torch.cat(axis_angle_distances[k]).cpu())
            mean_axis_angle_distances[k].append(torch.cat(axis_angle_distances[k]).cpu())
            

    for k in mean_axis_angle_distances.keys():
        mean_axis_angle_distances[k] = torch.vstack(mean_axis_angle_distances[k]).mean(axis=0)

    return mean_axis_angle_distances


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/pose/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{experiment_dir}/eval_{args.rot}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/modelnet40_normal_resampled/'
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal, subset=args.subset, class_choice=args.class_choice, align=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = int(args.subset[-2:])
    MODEL = importlib.import_module(args.model)
    
    pose_estimator = MODEL.get_model(args, normal_channel=args.normal, init_iden=args.init_iden).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    pose_estimator.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        mean_axis_angle_distances = test(pose_estimator.eval(), testDataLoader, num_class=num_class, num_rots=args.num_rots, num_partials=args.num_partials)
        for idx, (complete_error, partial_error) in enumerate(zip(mean_axis_angle_distances['complete'], mean_axis_angle_distances['partial'])):
            filename = TEST_DATASET.datapath[idx][1].split('/')[-1].split('.')[0]
            log_string(f'{idx} {filename} {complete_error.item():.6f} {partial_error.item():.6f}')



if __name__ == '__main__':
    args = parse_args()
    main(args)
