import os
import numpy as np
import time
from utils import cal_rel_pose

# transform poseGT [R|t] to [theta_x, theta_y, theta_z, x, y, z]
# save as .npy file
def create_pose_data(pose_dir):
    info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '04': [0, 270], '05': [0, 2760], '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]}
    start_t = time.time()
    for video in info.keys():
        fn = '{}{}.txt'.format(pose_dir, video)
        print('Transforming {}...'.format(fn))
        with open(fn) as f:
            lines = [line.split('\n')[0] for line in f.readlines()]
            poses = []
            for i in range(len(lines)):
                values = [float(value) for value in lines[i].split(' ')]
                if i > 0:
                    values_pre = [float(value) for value in lines[i - 1].split(' ')]
                    poses.append(cal_rel_pose(values_pre, values))
            poses = np.array(poses)
            base_fn = os.path.splitext(fn)[0]
            np.save(base_fn + '.npy', poses)
            print('Video {}: shape={}'.format(video, poses.shape))
    print('elapsed time = {}'.format(time.time() - start_t))


if __name__ == '__main__':
    create_pose_data('./data/poses/')
