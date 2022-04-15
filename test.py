# predicted as a batch
from model import DeepVIO
import numpy as np
from PIL import Image
import glob
import os
import time
import torch
import torchvision.transforms.functional as TF
import scipy.io as sio
import torch.nn as nn
from evaluation import kittiOdomEval
import argparse
from base_options import BaseOptions
from utils import data_partition, angle_to_R, pose_accu

torch.manual_seed(0)
np.random.seed(0)

opt = BaseOptions().parse()
VIONet = DeepVIO(opt)
VIONet.load_state_dict(torch.load(opt.load_path, map_location='cpu'))
VIONet.cuda()
VIONet.eval()

pose_eval = kittiOdomEval(opt)

for test_video in opt.test_list:
    
    df = data_partition(opt, test_video)
    hc = None
    answer = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], ]
    angle_out, trans_out, decision_out, probs_out = [], [], [], []

    for i, (image_seq, imu_seq, gt_seq) in enumerate(df):
        
        print('processing seq {}: {} / {}'.format(test_video, i, len(df)), end='\r', flush=True)
        
        x_in = image_seq.unsqueeze(0).cuda()
        i_in = imu_seq.unsqueeze(0).cuda()
        flag = True if i == 0 else False
        with torch.no_grad():
            angle, trans, decision, logits, hc = VIONet(x_in, i_in, is_first=flag, hc=hc)        
            probs = torch.nn.functional.softmax(logits, dim=-1)        

        angle = angle.squeeze(0).detach().cpu().numpy()
        trans = trans.squeeze(0).detach().cpu().numpy()
        pose_pred = np.hstack((angle, trans))

        # Accumulate the relative poses
        for index in range(angle.shape[0]):
            pose = pose_accu(answer[-1], pose_pred[index, :])
            answer.append(pose)

        angle_out.append(angle)
        trans_out.append(trans)
        decision_out.append(decision.squeeze(0).detach().cpu().numpy()[:, 0])
        probs_out.append(probs.squeeze(0).detach().cpu().numpy())
    
    ang_est = np.vstack(angle_out)
    trans_est = np.vstack(trans_out)
    dec_est = np.hstack(decision_out)    
    prob_est = np.vstack(probs_out)

    angle_rmse = np.sqrt(np.mean(np.sum((df.poses[:, :3]-ang_est)**2, -1)))
    trans_rmse = np.sqrt(np.mean(np.sum((df.poses[:, 3:]-trans_est)**2, -1)))
    
    # Save answer
    with open('{}/{}_pred.txt'.format(os.path.join(opt.save_dir, opt.model_name), test_video), 'w') as f:
        for pose in answer:
            if type(pose) == list:
                f.write(' '.join([str(r) for r in pose]))
            else:
                f.write(str(pose))
            f.write('\n')
    
    print('sequence: {}'.format(test_video))
    print('Average angle RMSE: {}'.format(angle_rmse))
    print('Average translation RMSE: {}'.format(trans_rmse))
    
    # Draw figures
    
    pose_eval.eval(test_video, dec_est, prob_est)   # set the value according to the predicted results


    
