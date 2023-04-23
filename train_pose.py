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
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model', default='pointnet_pose', help='Model name [default: %(default)s]',
                        choices = ['pointnet_pose', 'vn_pointnet_pose', 'dgcnn_pose', 'vn_dgcnn_pose'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size in training [default: 32]')
    parser.add_argument('--epoch', default=250, type=int, help='Number of epoch in training [default: 250]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate (for SGD it is multiplied by 100) [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Decay rate [default: 1e-4]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer for training [default: SGD]')
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
    parser.add_argument('--which_rot_metric', type=str, default='cosine', help='loss selection for rotation loss')
    parser.add_argument('--which_ortho_metric', type=str, default='MSE', help='loss selection for orthogonal loss')
#    parser.add_argument('--which_strict_rot', type=str, default='None', choices=['svd', 'gram_schmidt', 'None'], help='Define rotation tansform, [default: None]')
    parser.add_argument('--single_view_prob_train', default=0.0, type=float, help='Probability of single-view point cloud conversion for training [default: 0]')
    parser.add_argument('--single_view_prob_test', default=0.0, type=float, help='Probability of single-view point cloud conversion for testing [default: 0]')
    parser.add_argument('--renormalize', action='store_true', help='recenter and renormalize after partialization')
    parser.add_argument('--random_shift', action='store_true', help='apply random shifts to point cloud during training')
    return parser.parse_args()

def test(model, loader, acc_threshold):
    mean_correct = []
    mean_axis_angle_distances = []
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data

        trot = None
        if args.rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
        elif args.rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            if args.normal:
                points[:,:,0:3] = trot.transform_points(points[:,:,0:3])
                # Transform normals
                points[:,:,3:6] = trot.transform_normals(points[:,:,3:6])
            else:
                points = trot.transform_points(points)
                target_rot = trot.get_matrix()[:, :3, :3]
                
        if args.single_view_prob_test > 0:
            points = points.data.numpy()
            points, _ = provider.single_view_point_cloud(points, prob=args.single_view_prob_test, renormalize=args.renormalize)
            points = torch.Tensor(points)

        points = points.transpose(2, 1)
        points, target_rot = points.cuda(), target_rot.cuda()
        pose_estimator = model.eval()
        pred_rot = pose_estimator(points)
        pred_rot = provider.to_rotation_mat(pred_rot.detach())
        axis_angle_distances = provider.axis_angle_distance(pred_rot, target_rot)
        mean_axis_angle_distances.append(torch.mean(axis_angle_distances))
        correct = (axis_angle_distances <= acc_threshold).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    mean_axis_angle_distance = torch.mean(torch.tensor(mean_axis_angle_distances))
    instance_acc = np.mean(mean_correct)

    return mean_axis_angle_distance, instance_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath('pose')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string(f'Load dataset {args.subset}...')
    DATA_PATH = 'data/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal, subset=args.subset, class_choice=args.class_choice, align=True)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal, subset=args.subset, class_choice=args.class_choice, align=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = int(args.subset[-2:])
    MODEL = importlib.import_module(args.model)

    pose_estimator = MODEL.get_model(args, normal_channel=args.normal, init_iden=args.init_iden).cuda()
    criterion = MODEL.get_loss(which_rot_metric=args.which_rot_metric).cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        pose_estimator.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            pose_estimator.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            pose_estimator.parameters(),
            lr=args.learning_rate*100,
            momentum=0.9,
            weight_decay=args.decay_rate
        )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    mean_axis_angle_distances = []
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(points.shape[0]))
            if trot is not None:
                if args.normal:
                    points[:,:,0:3] = trot.transform_points(points[:,:,0:3])
                    # Transform normals
                    points[:,:,3:6] = trot.transform_normals(points[:,:,3:6])
                else:
                    points = trot.transform_points(points)
                target_rot = trot.get_matrix()[:, :3, :3]
            
            points = points.data.numpy()
            if args.single_view_prob_train > 0:
                points, _ = provider.single_view_point_cloud(points, prob=args.single_view_prob_train, renormalize=args.renormalize)
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:,0:3])
            if args.random_shift:
                if args.normal:
                    points = provider.shift_point_cloud(points, normal=True)
                else:
                    points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points, target_rot = points.cuda(), target_rot.cuda()

            optimizer.zero_grad()
            pose_estimator = pose_estimator.train()
            pred_rot = pose_estimator(points)
            loss = criterion(pred_rot, target_rot)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_rot = provider.to_rotation_mat(pred_rot.detach())
                axis_angle_distances = provider.axis_angle_distance(pred_rot, target_rot)
                mean_axis_angle_distances.append(torch.mean(axis_angle_distances))
                correct = (axis_angle_distances <= args.acc_threshold).cpu().sum()
                mean_correct.append(correct.item()/float(points.size()[0]))

            global_step += 1

        mean_axis_angle_distance = torch.mean(torch.tensor(mean_axis_angle_distances))
        instance_acc = np.mean(mean_correct)
        log_string(f'Train Mean Axis-Angle Distance: {mean_axis_angle_distance:.6f}')
        log_string(f'Train Instance Accuracy (<{args.acc_threshold:g}): {instance_acc:.6f}')

        with torch.no_grad():
            mean_axis_angle_distance, instance_acc = test(pose_estimator.eval(), testDataLoader, args.acc_threshold)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            log_string(f'Test Mean Axis-Angle Distance: {mean_axis_angle_distance:.6f}')
            log_string(f'Test Instance Accuracy (<{args.acc_threshold:g}): {instance_acc:.6f}')
            log_string(f'Best Instance Accuracy (<{args.acc_threshold:g}): {best_instance_acc:.6f}')

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'model_state_dict': pose_estimator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
