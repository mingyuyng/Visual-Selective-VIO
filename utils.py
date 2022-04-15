import os
import glob
import numpy as np
import time
import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF

class data_partition():
    def __init__(self, opt, folder):
        super(data_partition, self).__init__()
        self.opt = opt
        self.data_dir = opt.data_dir
        self.seq_len = opt.seq_len
        self.folder = folder
        self.load_data()

    def load_data(self):
        image_dir = self.data_dir + '/sequences/'
        imu_dir = self.data_dir + '/imus/'
        pose_dir = self.data_dir + '/poses/'

        self.img_paths = glob.glob('{}{}/image_2/*.png'.format(image_dir, self.folder))
        self.imus = sio.loadmat('{}{}.mat'.format(imu_dir, self.folder))['imu_data_interp']
        self.poses = np.load('{}{}.npy'.format(pose_dir, self.folder))
        self.img_paths.sort()
        
        img_paths_list, poses_list, imus_list = [], [], []
        start = 0
        n_frames = len(self.img_paths)
        while start + self.seq_len < n_frames:
            img_paths_list.append(self.img_paths[start:start+self.seq_len])
            poses_list.append(self.poses[start:start+self.seq_len-1])
            imus_list.append(self.imus[start*10:(start+self.seq_len-1)*10+1])
            start += self.seq_len - 1
        img_paths_list.append(self.img_paths[start:])
        poses_list.append(self.poses[start:])
        imus_list.append(self.imus[start*10:])

        self.imgs_path_p = np.asarray(img_paths_list)
        self.imus_p = np.asarray(imus_list)
        self.poses_p = np.asarray(poses_list)
    
    def __len__(self):
        return self.imgs_path_p.shape[0]

    def __getitem__(self, i):
        image_path_sequence = self.imgs_path_p[i]
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_img = TF.resize(img_as_img, size=(self.opt.img_h, self.opt.img_w))
            img_as_tensor = TF.to_tensor(img_as_img)-0.5
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        imu_sequence = torch.FloatTensor(self.imus_p[i])
        gt_sequence = self.poses_p[i][:, :6]
        return image_sequence, imu_sequence, gt_sequence


_EPS = np.finfo(float).eps * 4.0

# Determine whether the matrix R is a valid rotation matrix
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculate the rotation matrix from eular angles (roll, yaw, pitch)
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

# Calculate the eular angles (roll, yaw, pitch) from a rotation matrix
def euler_from_matrix(matrix):
    
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    
    cy = math.sqrt(M[0, 0]*M[0, 0] + M[1, 0]*M[1, 0])
    ay = math.atan2(-M[2, 0], cy)

    if ay < -math.pi/2 + _EPS and ay > -math.pi/2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2( -M[1, 2],  -M[0, 2])
    elif ay < math.pi/2 + _EPS and ay > math.pi/2 - _EPS:
        ax = 0
        az = math.atan2( M[1, 2],  M[0, 2])
    else:
        ax = math.atan2( M[2, 1],  M[2, 2])
        az = math.atan2( M[1, 0],  M[0, 0])

    return np.array([ax, ay, az])        
        
# Normalization angles to constrain that it is between -pi and pi   
def normalize_angle_delta(angle):
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

# Calculate the relative rotation (Eular angles) and translation from two pose matrices
# Rt1: a list of 12 floats
# Rt2: a list of 12 floats
def cal_rel_pose(Rt1, Rt2):
    Rt1 = np.reshape(np.array(Rt1), (3,4))
    Rt1 = np.concatenate((Rt1, np.array([[0,0,0,1]])), 0)
    Rt2 = np.reshape(np.array(Rt2), (3,4))
    Rt2 = np.concatenate((Rt2, np.array([[0,0,0,1]])), 0)
    
    # Calculate the relative transformation Rt_rel
    Rt1_inv = np.linalg.inv(Rt1)
    Rt_rel = Rt1_inv @ Rt2    

    R_rel = Rt_rel[:3, :3]
    t_rel = Rt_rel[:3, 3]
    assert(isRotationMatrix(R_rel))
    
    # Extract the Eular angle from the relative rotation matrix
    x, y, z = euler_from_matrix(R_rel)
    theta = [x, y, z]
    
    pose_rel = np.concatenate((theta, t_rel))
    return pose_rel

# Calculate the 3x4 transformation matrix from Eular angles and translation vector
# pose: (3 angles, 3 translations) 
def angle_to_R(pose):
    R = eulerAnglesToRotationMatrix(pose[:3])
    t = pose[3:].reshape(3, 1)    
    R = np.concatenate((R, t), 1)
    return R

# Accumulate the pose from previous pose and the relative pose
# R_pre: a list of 12 floats
# R_rel: a list of 12 floats
def pose_accu(R_pre, R_rel):
    poses_pre = np.array(R_pre).reshape(3, 4)
    R_pre = poses_pre[:, :3]
    t_pre = poses_pre[:, 3]

    Rt_rel = angle_to_R(R_rel)
    R_rel = Rt_rel[:, :3]
    t_rel = Rt_rel[:, 3]

    R = R_pre @ R_rel
    t = R_pre.dot(t_rel) + t_pre
    return np.concatenate((R, t.reshape(3, 1)), 1).flatten().tolist()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w
