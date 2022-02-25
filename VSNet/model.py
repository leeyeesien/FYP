from cv2 import QT_NEW_BUTTONBAR, norm
import torch
import torch.nn as nn
import torchvision.models as models
import transformations as T
import numpy as np
import roma

def loss_quat(output, target):
    # compute rmse for translation
    trans_rmse = torch.sqrt(torch.mean((output[:, :3] - target[:, :3]) ** 2, dim=1))

    return torch.mean(trans_rmse)


def combined_loss_quat(device, output, target, weights=[1 / 2, 1 / 2]):
    # compute rmse for translation
    trans_rmse = torch.sqrt(torch.mean((output[:, :3] - target[:, :3]) ** 2, dim=1))

    # normalize quaternions
    normalized_quat = output[:, 3:] / torch.sqrt(torch.sum(output[:, 3:] ** 2, dim=1, keepdim=True))
    
    # new quat error
    rot_errors_torch_list = []
    for i in range(normalized_quat.shape[0]):
        q2=target[i, 3:]
        q2_xyzw = torch.empty(4)
        q2_xyzw[0] = q2[1]
        q2_xyzw[1] = q2[2]
        q2_xyzw[2] = q2[3]
        q2_xyzw[3] = q2[0]

        if torch.dot(normalized_quat[i], q2) < 0:
            normalized_quat[i] = -normalized_quat[i]

        normalized_quat_xyzw = torch.empty(4)
        normalized_quat_xyzw[0] = normalized_quat[i][1]
        normalized_quat_xyzw[1] = normalized_quat[i][2]    
        normalized_quat_xyzw[2] = normalized_quat[i][3]    
        normalized_quat_xyzw[3] = normalized_quat[i][0]    

        quat_inv_xyzw=roma.quat_inverse(normalized_quat_xyzw)
        quat_error_xyzw=roma.quat_product(q2_xyzw, quat_inv_xyzw)

        quat_error_wxyz = torch.empty(4)
        quat_error_wxyz[0] = quat_error_xyzw[3]
        quat_error_wxyz[1] = quat_error_xyzw[0]
        quat_error_wxyz[2] = quat_error_xyzw[1]
        quat_error_wxyz[3] = quat_error_xyzw[2]

        if quat_error_wxyz[0] > 1.0:
            quat_error_wxyz[0] = 1.0
        elif quat_error_wxyz[0] < -1.0:
            quat_error_wxyz[0] = -1.0

        den = torch.sqrt(1.0 - quat_error_wxyz[0] * quat_error_wxyz[0])
        zero_tensor = torch.empty(1)
        if torch.isclose(den, zero_tensor):
            rot_errors_torch = torch.zeros(3)
        else:
            rot_errors_torch = (quat_error_wxyz[1:] * 2.0 * torch.acos(quat_error_wxyz[0])) / den

        rot_errors_torch_list.append(rot_errors_torch)
        rot_errors_tensor = torch.stack(rot_errors_torch_list)

        # compute rmse for rotation
        quat_rmse = torch.sqrt(torch.mean((rot_errors_tensor) ** 2, dim=1))
    
    quat_rmse = quat_rmse.to(device)
    return torch.mean(weights[0] * trans_rmse + weights[1] * quat_rmse)


def tf_dist_loss(output, target, device, unit_length=0.1):
    loss = torch.zeros([output.shape[0], 1])
    unit_vec = torch.tensor([unit_length, unit_length, unit_length, 0.0]).view(4, 1).to(device)
    add_row = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 4).to(device)

    for i in range(output.shape[0]):
        output_slice = output[0].view(3, 4)
        output_mat = torch.cat((output_slice, add_row), dim=0)

        transformed = torch.matmul(torch.matmul(target[i, :].inverse(), output_mat), unit_vec)

        loss[i] = torch.sqrt(torch.sum((transformed[:3] - unit_vec[:3]) ** 2))

    return torch.mean(loss)

class VSNet(nn.Module):

    def __init__(self, num_classes=2):
        super(VSNet, self).__init__()
        self.caffenet = models.alexnet(pretrained=True)

        self.caffenet.classifier[1] = nn.Linear(14 * 19 * 96 * 2, 4096)
        self.caffenet.classifier[-1] = nn.Linear(4096, 1024)
        self.channelRed = nn.Conv2d(256, 96, 1)
        self.output = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, num_classes))

    def weight_init(self):
        for mod2 in self.output:
            if isinstance(mod2, nn.Conv2d) or isinstance(mod2, nn.Linear):
                torch.nn.init.xavier_uniform_(mod2.weight)
        torch.nn.init.xavier_uniform_(self.channelRed.weight)
        torch.nn.init.xavier_uniform_(self.caffenet.classifier[1].weight)
        torch.nn.init.xavier_uniform_(self.caffenet.classifier[-1].weight)

    def forward(self, a, b):
        a = self.caffenet.features(a)
        a = self.channelRed(a)
        a = a.view(a.size(0), 14 * 19 * 96)

        b = self.caffenet.features(b)
        b = self.channelRed(b)
        b = b.view(b.size(0), 14 * 19 * 96)

        concat = torch.cat((a, b), 1)
        match = self.caffenet.classifier(concat)
        match = self.output(match)

        return match




