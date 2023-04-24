import numpy as np
import open3d as o3d
import torch
from data_utils.ModelNetDataLoader import pc_normalize
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

EPS = 1e-10

def batched_trace(mat):
    return mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

def axis_angle_distance(R1, R2):
    M = torch.matmul(R1, R2.transpose(1, 2))
    dist = torch.acos(torch.clamp((batched_trace(M) - 1) / 2., -1 + EPS, 1 - EPS))
    dist = (180 / np.pi) * dist
    return dist

def to_rotation_mat(rot, which_rot='svd'):
    if which_rot == 'svd':
        u, s, v = torch.svd(rot)
        M_TM_pow_minus_half = torch.matmul(v / (s + EPS).unsqueeze(1), v.transpose(2, 1))
        rot_mat = torch.matmul(rot, M_TM_pow_minus_half)
        # If gradient trick is rqeuired:
        #rot_mat = (rot_mat - rot).detach() + rot
    else:
        # Gram–Schmidt
        rot_vec0 = rot[:,0,:]
        rot_vec1 = rot[:,1,:] - rot_vec0 * torch.sum(rot_vec0 *  rot[:,1,:], dim=-1, keepdim=True) \
                           / (torch.sum(rot_vec0 **2, dim=-1, keepdim=True) + EPS)

        rot_vec2 = rot[:,2,:] - rot_vec0 * torch.sum(rot_vec0 *  rot[:,2,:], dim=-1, keepdim=True) \
                           / (torch.sum(rot_vec0 **2, dim=-1, keepdim=True) + EPS)
        rot_vec2 = rot_vec2 - rot_vec1 * torch.sum(rot_vec1 * rot[:, 2, :], dim=-1, keepdim=True) \
                           / (torch.sum(rot_vec1 **2, dim=-1, keepdim=True) + EPS)
        rot_mat = torch.stack([rot_vec0, rot_vec1, rot_vec2], dim=1)
        rot_mat = rot_mat / torch.sqrt((torch.sum(rot_mat ** 2, dim=2, keepdim=True) + EPS))
    return rot_mat

def partialize_point_cloud(batch_data, prob=0.5, camera_direction='random', renormalize=False):
    """ Randomly convert complete point cloud to single view to augument the dataset
        Uses Open3D "Hidden Point Removal" algorithm
        (http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Hidden-point-removal)
        Conversion is per shape with camera placed along +z axis, works with and without normals
        Input:
          batch_data: BxNx3 (or BxNx6) array, original batch of point clouds (and optional normals)
          prob: per-shape probability of single-view point cloud conversion
          camera_direction: 'random' or np.ndarray for specific direction
        Return:
          processed_data: BxNx3 (or BxNx6) array, processed batch of point clouds (and optional normals)
          partialized: length B array, boolean flag for which point clouds were converted to single view
    """
    # Initialize processed_data as a copy of batch_data
    processed_data = batch_data.copy()
    batch_size = batch_data.shape[0]
    
    # Compute camera directions for each point cloud in batch
    if camera_direction == 'random':
        camera_direction = (random_rotations(batch_size) @ np.array([0, 0, 1])).numpy()
    elif isinstance(camera_direction, (list, np.ndarray)):
        camera_direction = np.asarray(camera_direction, dtype=float).reshape(-1, 3)
        if len(camera_direction) == 1:
            camera_direction = np.tile(camera_direction, (batch_size, 1))
        elif len(camera_direction) != batch_size:
            raise ValueError('number of camera directions must equal 1 or batch_size')
    camera_direction /= np.linalg.norm(camera_direction, axis=1, keepdims=True)
    

    partialized = np.random.uniform(size=batch_size) < prob
    for k in np.argwhere(partialized).ravel():

        # Apply HPR operator from specified direction
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(batch_data[k, :, 0:3])
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        _, pt_map = pcd.hidden_point_removal(camera_direction[k] * diameter, diameter * 100)
        if o3d.__version__ == '0.9.0.0':
            points = np.asarray(pcd.select_down_sample(pt_map).points)
        else:
            points = np.asarray(pcd.select_by_index(pt_map).points)

        if renormalize:
            points = pc_normalize(points)

        # Concatenate matching normals if they exist
        points = np.concatenate([points, batch_data[k, pt_map, 3:6]], axis=-1)

        # Place points in processed data array, padding with the first point
        processed_data[k, :len(pt_map), :] = points
        processed_data[k, len(pt_map):, :] = points[0]

    info = dict(camera_direction=camera_direction, partialized=partialized)
    return processed_data, info

def single_view_point_cloud(batch_data, prob=0.5, renormalize=False):
    """ Randomly convert point cloud to single view to augument the dataset
        Uses Open3D "Hidden Point Removal" algorithm (http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Hidden-point-removal) 
        Conversion is per shape with camera placed along +z axis, works with and without normals
        Input:
          batch_data: BxNx3 (or BxNx6) array, original batch of point clouds (and optional normals)
          prob: per-shape probability of single-view point cloud conversion
        Return:
          processed_data: BxNx3 (or BxNx6) array, processed batch of point clouds (and optional normals)
          single_view: length B array, boolean flag for which point clouds were converted to single view
    """
    processed_data = batch_data.copy()
    single_view = np.random.uniform(size=batch_data.shape[0]) < prob
    for k in np.argwhere(single_view).ravel():
        # Convert point cloud to single view
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(batch_data[k, :, 0:3])
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        camera = [0, 0, diameter]
        radius = diameter * 100
        _, pt_map = pcd.hidden_point_removal(camera, radius)
        if o3d.__version__ == '0.9.0.0':
            points = np.asarray(pcd.select_down_sample(pt_map).points)
        else:
            points = np.asarray(pcd.select_by_index(pt_map).points)
        # Renormalize single-view point cloud
        if renormalize:
            points = pc_normalize(points)
        # Concatenate matching normals if they exist
        points = np.concatenate([points, batch_data[k, pt_map, 3:6]], axis=-1)
        # Place points in processed data array, padding with the first point
        n = len(pt_map)
        processed_data[k, :n, :] = points
        processed_data[k, n:, :] = points[0]
    return processed_data, single_view


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:,idx,:]

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k,:,0:3]
        shape_normal = batch_xyz_normal[k,:,3:6]
        batch_xyz_normal[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal

def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1,3)), rotation_matrix)
    return rotated_data



def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def shift_point_cloud(batch_data, shift_range=0.1, normal=False):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        if normal:
            batch_data[batch_index,:,0:3] += shifts[batch_index,:]
            # Shift normals
            batch_data[batch_index,:,3:6] += shifts[batch_index,:]
        else:
            batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc



