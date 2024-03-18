import torch
import torch.nn as nn

from .common import chamfer_dist

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args

    def loss_on_cd_points(self, align_viewpoints): # view_num batch*3*point_num
        output = 0
        for i in range(1, self.args.view_num):
            dist1, dist2 = chamfer_dist(align_viewpoints[i-1].transpose(1, 2), align_viewpoints[i].transpose(1, 2)) # batch*3*point_num--->batch*point_num
            dist1, _ = (-1*dist1).topk(k=self.args.loss_top_k, dim=1) # batch*topk
            dist2, _ = (-1*dist2).topk(k=self.args.loss_top_k, dim=1)
            dist1 = -1 * dist1
            dist2 = -1 * dist2
            output += torch.mean(dist1 + dist2) * 0.5  # average distance of point to point

        return output / (self.args.view_num-1)

    def loss_on_cd_center(self, rotary_center, offset_rotary_center):
        output = (((rotary_center-offset_rotary_center)**2).sum(-1))**0.5
        return torch.mean(output)

    def forward(self, weight_list, rotary_center_list, modified_rotary_center_list, align_viewpoints_list, Rotary_Center_tensor_batch):
        zero_tensor = torch.mean(torch.zeros((1), dtype=torch.float)).cuda()

        iter_time = len(weight_list)
        loss = torch.zeros((1), dtype=torch.float).cuda()
        loss_stages = []

        if self.args.weight_cd_points > 0:
            loss_cd_points = 0
            for i in range(iter_time):
                loss_cd_points += self.args.gamma ** (iter_time - i - 1) * self.loss_on_cd_points(align_viewpoints_list[i])
            loss += self.args.weight_cd_points*loss_cd_points/iter_time
            loss_stages.append(loss_cd_points/iter_time)
        else:
            loss_stages.append(zero_tensor)

        if self.args.weight_cd_modified_center > 0:
            loss_cd_modified_center = 0
            for i in range(iter_time):
                loss_cd_modified_center += self.args.gamma ** (iter_time - i - 1) * self.loss_on_cd_center(Rotary_Center_tensor_batch, modified_rotary_center_list[i])
            loss += self.args.weight_cd_modified_center*loss_cd_modified_center/iter_time
            loss_stages.append(loss_cd_modified_center/iter_time)
        else:
            loss_stages.append(zero_tensor)

        return loss, loss_stages