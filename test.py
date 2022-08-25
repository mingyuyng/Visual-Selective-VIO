from model import DeepVIO
import numpy as np
import os
import torch
from options import BaseOptions
from utils import data_partition, kitti_eva

def init_net(opt, net):
    net.load_state_dict(torch.load(opt.load_path, map_location='cpu'))
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(opt.gpu_ids[0])
        net = torch.nn.DataParallel(net, opt.gpu_ids)  # multi-GPUs
    return net

def test_one_path(opt, test_path, net):
    df = data_partition(opt, test_path)
    hc = None
    pose_list, decision_list, probs_list= [], [], []
    for i, (image_seq, imu_seq, gt_seq) in enumerate(df):
        print('processing seq {}: {} / {}'.format(test_path, i, len(df)), end='\r', flush=True)
        x_in = image_seq.unsqueeze(0).cuda()
        i_in = imu_seq.unsqueeze(0).cuda()
        with torch.no_grad():
            pose, decision, probs, hc = net(x_in, i_in, is_first=(i==0), hc=hc)
        pose_list.append(pose.squeeze(0).detach().cpu().numpy())
        decision_list.append(decision.squeeze(0).detach().cpu().numpy()[:, 0])
        probs_list.append(probs.squeeze(0).detach().cpu().numpy())
    pose_est = np.vstack(pose_list)
    dec_est = np.hstack(decision_list)
    prob_est = np.vstack(probs_list)
    kitti_eva(opt, test_path, pose_est, dec_est, prob_est, df.poses)
    print('*'*15)

def main():
    # Load the options, model, and evaluator
    opt = BaseOptions().parse()
    VIONet = DeepVIO(opt)
    VIONet = init_net(opt, VIONet).eval()

    # Set the random seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # Evaluate each test video
    for test_video in opt.test_list:
        test_one_path(opt, test_video, VIONet)

if __name__ == '__main__':
    main()