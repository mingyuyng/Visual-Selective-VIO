import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )

# The inertial encoder for raw imu data
class Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_encoder, self).__init__()
        
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))
        self.proj = nn.Linear(256 * 1 * 11, opt.imu_f_len)

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
        x = self.encoder_conv(x.permute(0, 2, 1))                 # x: (N x seq_len, 64, 11)
        out = self.proj(x.view(x.shape[0], -1))                   # out: (N x seq_len, 256)
        return out.view(batch_size, seq_len, 256)

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        # CNN
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.visual_f_len)
        self.inertial_encoder = Inertial_encoder(opt)

    def forward(self, img, imu):
        # x: (batch, seq_len, channel, width, height)
        # stack_image

        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        # CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v_high = self.encode_image(v)
        v_high = v_high.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v_high = self.visual_head(v_high)  # (batch, seq_len, 256)

        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)

        return v_high, imu

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


# The fusion module
class Fusion_module(nn.Module):
    def __init__(self, opt, temp=None):
        super(Fusion_module, self).__init__()
        self.fuse_method = opt.fuse_method
        self.f_len = opt.imu_f_len + opt.visual_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len))
            if temp is None:
                self.temp = 1
            else:
                self.temp = temp

    def forward(self, v, i):
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]

# The policy network module
class PolicyNet(nn.Module):
    def __init__(self, opt):
        super(PolicyNet, self).__init__()
        in_dim = opt.rnn_hidden_size + opt.imu_f_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2))

    def forward(self, x, temp):
        logits = self.net(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return logits, hard_mask

# The pose estimation network
class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()

        # The main RNN network
        f_len = opt.visual_f_len + opt.imu_f_len
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True)

        self.fuse = Fusion_module(opt)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, fv, fv_alter, fi, dec, prev=None):

        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())

        batch_size = fv.shape[0]
        seq_len = fi.shape[1]

        if fv_alter is None:
            v_in = fv
        else:
            v_in = fv * dec[:, :, 0].unsqueeze(-1) + fv_alter * dec[:, :, 1].unsqueeze(-1)

        fused = self.fuse(v_in, fi)
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)
        angle = pose[:, :, :3]
        trans = pose[:, :, 3:]

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return angle, trans, hc



class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()

        self.Feature_net = Encoder(opt)
        self.Pose_net = Pose_RNN(opt)
        self.Policy_net = PolicyNet(opt)
        self.opt = opt

    def forward(self, img, imu, is_first=True, hc=None, temp=5):

        fv, fi = self.Feature_net(img, imu)
        batch_size = fv.shape[0]
        seq_len = fv.shape[1]

        angle_list, trans_list, decision_list, logit_list = [], [], [], []
        hidden = torch.zeros(batch_size, self.opt.rnn_hidden_size).to(fv.device) if hc is None else hc[0].contiguous()[:, -1, :]
        fv_alter = torch.zeros_like(fv)

        for i in range(seq_len):
            if i == 0 and is_first:
                # The first relative pose is estimated by both images and imu by default
                angle, trans, hc = self.Pose_net(fv[:, i, :].unsqueeze(1), None, fi[:, i, :].unsqueeze(1), None, hc)
            else:
                # Otherwise, sample the decision from the policy network
                p_in = torch.cat((fi[:, i, :], hidden), -1)
                logits, decision = self.Policy_net(p_in.detach(), temp)

                # prob = torch.nn.functional.softmax(logits, dim=-1)
                # weight = torch.rand_like(prob[:,0])  
                # decision = (weight < prob[:,0]).to(torch.float32)
                # decision = decision.unsqueeze(1).unsqueeze(1)
                # decision = torch.cat((decision, 1-decision), -1)

                decision = decision.unsqueeze(1)                   
                angle, trans, hc = self.Pose_net(fv[:, i, :].unsqueeze(1), fv_alter[:, i, :].unsqueeze(1), fi[:, i, :].unsqueeze(1), decision, hc)
                decision_list.append(decision) 
                logit_list.append(logits) 
            
            angle_list.append(angle)
            trans_list.append(trans)
            hidden = hc[0].contiguous()[:, -1, :]

        angles = torch.cat(angle_list, dim=1)
        trans = torch.cat(trans_list, dim=1)
        decision = torch.cat(decision_list, dim=1)
        logit = torch.cat(logit_list, dim=0)

        return angles, trans, decision, logit, hc
