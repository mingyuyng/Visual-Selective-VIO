
#
# Copyright Qing Li (hello.qingli@gmail.com) 2018. All Rights Reserved.
#
# References: 1. KITTI odometry development kit: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
#             2. A Geiger, P Lenz, R Urtasun. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR 2012.
#

import glob
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
# choose other backend that not required GUI (Agg, Cairo, PS, PDF or SVG) when use matplotlib
plt.switch_backend('agg')
import matplotlib.backends.backend_pdf
from utils import moving_average

class kittiOdomEval():
    def __init__(self, opt):

        self.gt_dir = opt.data_dir + '/poses'
        assert os.path.exists(self.gt_dir), "Error of ground_truth pose path!"
        self.lengths = [100,200,300,400,500,600,700,800]
        self.num_lengths = len(self.lengths)
        self.result_dir = os.path.join(opt.save_dir, opt.model_name)
        self.opt = opt

    def loadPoses(self, file_name):
        '''
            Each line in the file should follow one of the following structures
            (1) idx pose(3x4 matrix in terms of 12 numbers)
            (2) pose(3x4 matrix in terms of 12 numbers)
        '''
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        frame_idx = 0
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split()]
            withIdx = int(len(line_split)==13)
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row*4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            poses[frame_idx] = P
        return poses

    def trajectoryDistances(self, poses):
        '''
            Compute the length of the trajectory
            poses dictionary: [frame_idx: pose]
        '''
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx)-1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i+1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0,3] - P2[0,3]
            dy = P1[1,3] - P2[1,3]
            dz = P1[2,3] - P2[2,3]
            dist.append(dist[i]+np.sqrt(dx**2+dy**2+dz**2))
        self.distance = dist[-1]
        return dist

    def rotationError(self, pose_error):
        a = pose_error[0,0]
        b = pose_error[1,1]
        c = pose_error[2,2]
        d = 0.5*(a+b+c-1.0)
        return np.arccos(max(min(d,1.0),-1.0))

    def translationError(self, pose_error):
        dx = pose_error[0,3]
        dy = pose_error[1,3]
        dz = pose_error[2,3]
        return np.sqrt(dx**2+dy**2+dz**2)

    def lastFrameFromSegmentLength(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1

    def calcSequenceErrors(self, poses_gt, poses_result):
        err = []
        self.max_speed = 0
        # pre-compute distances (from ground truth as reference)
        dist = self.trajectoryDistances(poses_gt)
        # every second, kitti data 10Hz
        self.step_size = 10
        # for all start positions do
        # for first_frame in range(9, len(poses_gt), self.step_size):
        for first_frame in range(0, len(poses_gt), self.step_size):
            # for all segment lengths do
            for i in range(self.num_lengths):
                # current length
                len_ = self.lengths[i]
                # compute last frame of the segment
                last_frame = self.lastFrameFromSegmentLength(dist, first_frame, len_)

                # Continue if sequence not long enough
                if last_frame == -1 or not(last_frame in poses_result.keys()) or not(first_frame in poses_result.keys()):
                    continue

                # compute rotational and translational errors, relative pose error (RPE)
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotationError(pose_error)
                t_err = self.translationError(pose_error)

                # compute speed
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1*num_frames)   # 10Hz
                if speed > self.max_speed:
                    self.max_speed = speed
                err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
        return err

    def calcSpeed(self, poses_gt, poses_result):
        speed_err_gt_list = []
        speed_err_est_list = []
        # for all start positions do
        for first_frame in range(len(poses_gt)-1):
            # compute rotational and translational errors, relative pose error (RPE)
            pose_error_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[first_frame+1])
            pose_error_est = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[first_frame+1])
            speed_err_gt = self.translationError(pose_error_gt)
            speed_err_est = self.translationError(pose_error_est)
            speed_err_gt_list.append(speed_err_gt)
            speed_err_est_list.append(speed_err_est)
        return speed_err_gt_list, speed_err_est_list


    def saveSequenceErrors(self, err, file_name):
        fp = open(file_name,'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write+"\n")
        fp.close()

    def computeOverallErr(self, seq_err):
        t_err = 0
        r_err = 0
        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err

    def plotPath_2D(self, seq, poses_gt, poses_result, plot_path_dir, decision, speed):

        fontsize_ = 10
        plot_keys = ["Ground Truth", "Ours"]
        start_point = [0, 0]
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'

        ### get the value
        if poses_gt:
            poses_gt = [(k,poses_gt[k]) for k in sorted(poses_gt.keys())]
            x_gt = np.asarray([pose[0,3] for _,pose in poses_gt])
            y_gt = np.asarray([pose[1,3] for _,pose in poses_gt])
            z_gt = np.asarray([pose[2,3] for _,pose in poses_gt])
        poses_result = [(k,poses_result[k]) for k in sorted(poses_result.keys())]
        x_pred = np.asarray([pose[0,3] for _,pose in poses_result])
        y_pred = np.asarray([pose[1,3] for _,pose in poses_result])
        z_pred = np.asarray([pose[2,3] for _,pose in poses_result])

        fig = plt.figure(figsize=(20,6), dpi=100)
        ### plot the figure
        plt.subplot(1,3,1)
        ax = plt.gca()
        if poses_gt: plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
        plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        ### set the range of x and y
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        plot_radius = max([abs(lim - mean_)
                            for lims, mean_ in ((xlim, xmean),
                                                (ylim, ymean))
                            for lim in lims])
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        plt.subplot(1,3,2)
        ax = plt.gca()
        cout = np.insert(decision, 0, 0) * 100
        cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
        cbar = fig.colorbar(cax, ticks=[0, 5, 10, 15, 20])
        cbar.ax.set_yticklabels(['0%', '5%', '10%', '15%', '20%'])


        plt.subplot(1,3,3)
        ax = plt.gca()
        cout = np.insert(speed, 0, 0) * 10
        cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
        cbar = fig.colorbar(cax, ticks=[0.1, 4, 8, 12])
        cbar.ax.set_yticklabels(['0m/s', '4m/s', '8m/s', '12m/s'])


        png_title = "{}_path".format(seq)
        plt.savefig(plot_path_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        plt.close()


    def plotpolicy(self, seq, plot_path_dir, decision, prob):

        fontsize_ = 10
        plot_keys = ["probability", "decisions"]
        start_point = [0, 0]
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'


        fig = plt.figure(figsize=(20,6), dpi=100)
        ### plot the figure
        ax = plt.gca()

        plt.plot(prob, style_gt, label=plot_keys[0])
        plt.stem(decision, label=plot_keys[1])
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('index', fontsize=fontsize_)
        ### set the range of x and y

        png_title = "{}_prob".format(seq)
        plt.savefig(plot_path_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        plt.close()


        fig = plt.figure(figsize=(20,6), dpi=100)
        ### plot the figure
        ax = plt.gca()

        plt.plot(prob, 'rx-', label=plot_keys[0])
        plt.stem(decision, label=plot_keys[1])
        plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('index', fontsize=fontsize_)
        ax.set_xlim([0, 200])
        ### set the range of x and y

        png_title = "{}_prob_zoom".format(seq)
        plt.savefig(plot_path_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        plt.close()


    def computeSegmentErr(self, seq_errs):
        '''
            This function calculates average errors for different segment.
        '''
        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []

        # Get errors
        for err in seq_errs:
            len_  = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])

        # Compute average
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:,0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:,1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def computeSpeedErr(self, seq_errs):
        '''
            This function calculates average errors for different speed.
        '''
        segment_errs = {}
        avg_segment_errs = {}
        for s in range(2, 25, 2):
            segment_errs[s] = []

        # Get errors
        for err in seq_errs:
            speed = err[4]
            t_err = err[2]
            r_err = err[1]
            for key in segment_errs.keys():
                if np.abs(speed - key) < 2.0:
                    segment_errs[key].append([t_err, r_err])

        # Compute average
        for key in segment_errs.keys():
            if segment_errs[key] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[key])[:,0])
                avg_r_err = np.mean(np.asarray(segment_errs[key])[:,1])
                avg_segment_errs[key] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[key] = []
        return avg_segment_errs

    def call_evo_traj(self, pred_file, save_file, gt_file=None, plot_plane='xy'):
        command = ''
        if os.path.exists(save_file): os.remove(save_file)

        if gt_file != None:
            command = ("evo_traj kitti %s --ref=%s --plot_mode=%s --save_plot=%s") \
                        % (pred_file, gt_file, plot_plane, save_file)
        else:
            command = ("evo_traj kitti %s --plot_mode=%s --save_plot=%s") \
                        % (pred_file, plot_plane, save_file)
        os.system(command)

    def eval(self, seq, decision, prob):
        '''
            to_camera_coord: whether the predicted pose needs to be convert to camera coordinate
        '''

        decision = np.insert(decision, 0, 1)
        prob = np.insert(prob[:,0], 0, 1)
        decision_smooth = moving_average(decision , self.opt.window_size)
        eval_dir = self.result_dir

        if not os.path.exists(eval_dir): os.makedirs(eval_dir)
        total_err = []
        ave_errs = {}

        eva_seq_dir = os.path.join(eval_dir, '{}_eval'.format(seq))
        pred_file_name = self.result_dir + '/{}_pred.txt'.format(seq)
        gt_file_name   = self.gt_dir + '/{}.txt'.format(seq)
        assert os.path.exists(pred_file_name), "File path error: {}".format(pred_file_name)
        if not os.path.exists(eva_seq_dir): os.makedirs(eva_seq_dir)

        poses_result = self.loadPoses(pred_file_name)
        poses_gt = self.loadPoses(gt_file_name)

        # ----------------------------------------------------------------------
        # compute sequence errors
        seq_err = self.calcSequenceErrors(poses_gt, poses_result)
        self.saveSequenceErrors(seq_err, eva_seq_dir + '/{}_error.txt'.format(seq))
        total_err += seq_err
        speed_gt, speed_est = self.calcSpeed(poses_gt, poses_result)

        # ----------------------------------------------------------------------
        # Compute segment errors
        avg_segment_errs = self.computeSegmentErr(seq_err)
        avg_speed_errs   = self.computeSpeedErr(seq_err)

        # ----------------------------------------------------------------------
        # compute overall error
        ave_t_err, ave_r_err = self.computeOverallErr(seq_err)

        print ("\nSequence: " + str(seq))
        print ('Distance (m): %d' % self.distance)
        print ('Max speed (km/h): %d' % (self.max_speed*3.6))
        print ("Average sequence translational RMSE (%):   {0:.4f}".format(ave_t_err * 100))
        print ("Average sequence rotational error (deg/m): {0:.4f}\n".format(ave_r_err/np.pi * 180))

        with open(eva_seq_dir + '/%s_stats.txt' % seq, 'w') as f:
            f.writelines('Average sequence translation RMSE (%):    {0:.4f}\n'.format(ave_t_err * 100))
            f.writelines('Average sequence rotation error (deg/m):  {0:.4f}'.format(ave_r_err/np.pi * 180))

        # ----------------------------------------------------------------------
        # Ploting
        self.plotPath_2D(seq, poses_gt, poses_result, eva_seq_dir, decision_smooth, speed_gt)
        self.plotpolicy(seq, eva_seq_dir, decision, prob)
        plt.close('all')


