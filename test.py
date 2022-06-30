# predicted as a batch
from model import DeepVIO
import numpy as np
import os
import torch
from options import BaseOptions
from utils import data_partition, kitti_eva

# Load the options, model, and evaluator
opt = BaseOptions().parse()
VIONet = DeepVIO(opt)
VIONet.load_state_dict(torch.load(opt.load_path, map_location='cpu'))
VIONet.cuda()
VIONet.eval()

# Set the random seed
torch.manual_seed(0)
np.random.seed(0)


# Evaluate each test video
for test_video in opt.test_list:

    # Break the whole video into short sequences
    df = data_partition(opt, test_video)
    hc = None
    pose_list, decision_list, probs_list= [], [], []

    for i, (image_seq, imu_seq, gt_seq) in enumerate(df):
        
        print('processing seq {}: {} / {}'.format(test_video, i, len(df)), end='\r', flush=True)

        x_in = image_seq.unsqueeze(0).cuda()
        i_in = imu_seq.unsqueeze(0).cuda()
        flag = True if i == 0 else False
        with torch.no_grad():
            pose, decision, probs, hc = VIONet(x_in, i_in, is_first=flag, hc=hc)
        
        pose_list.append(pose.squeeze(0).detach().cpu().numpy())
        decision_list.append(decision.squeeze(0).detach().cpu().numpy()[:, 0])
        probs_list.append(probs.squeeze(0).detach().cpu().numpy())

    pose_est = np.vstack(pose_list)
    dec_est = np.hstack(decision_list)
    prob_est = np.vstack(probs_list)
    
    kitti_eva(opt, test_video, pose_est, dec_est, prob_est, df.poses)