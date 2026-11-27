import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


N_PARAMS = {'affine': 6,
            'translation': 2,
            'rotation': 1,
            'scale': 2,
            'shear': 2,
            'rotation_scale': 3,
            'translation_scale': 4,
            'rotation_translation': 3,
            'rotation_translation_scale': 5}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class STNModule(nn.Module):
    def __init__(self, in_num, padding=0, mode='rotation'):
        super(STNModule, self).__init__()

        self.feat_size = (224 + 2 * padding) // 4
        self.padding = padding
        self.stn_mode = mode
        self.stn_n_params = N_PARAMS[self.stn_mode]

        self.conv = nn.Sequential(
            conv3x3(in_planes=in_num, out_planes=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv3x3(in_planes=64, out_planes=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * self.feat_size * self.feat_size, out_features=1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=self.stn_n_params),
        )
        self.fc[2].weight.data.fill_(0)
        self.fc[2].weight.data.zero_()

        if self.stn_mode == 'affine':
            self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif self.stn_mode in ['translation', 'shear']:
            self.fc[2].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        elif self.stn_mode == 'scale':
            self.fc[2].bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation':
            self.fc[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        elif self.stn_mode == 'rotation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'translation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))

    def forward(self, x):
        mode = self.stn_mode
        batch_size = x.size(0)
        conv_x = self.conv(x)
        theta = self.fc(conv_x.view(batch_size, -1))

        if mode == 'affine':
            theta1 = theta.view(batch_size, 2, 3)
        else:
            theta1 = Variable(torch.zeros([batch_size, 2, 3], dtype=torch.float32, device=x.device),
                              requires_grad=True)
            theta1 = theta1 + 0
            theta1[:, 0, 0] = 1.0
            theta1[:, 1, 1] = 1.0
            if mode == 'translation':
                theta1[:, 0, 2] = theta[:, 0]
                theta1[:, 1, 2] = theta[:, 1]
            elif mode == 'rotation':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle)
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle)
            elif mode == 'scale':
                theta1[:, 0, 0] = theta[:, 0]
                theta1[:, 1, 1] = theta[:, 1]
            elif mode == 'shear':
                theta1[:, 0, 1] = theta[:, 0]
                theta1[:, 1, 0] = theta[:, 1]
            elif mode == 'rotation_scale':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle) * theta[:, 1]
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle) * theta[:, 2]
            elif mode == 'translation_scale':
                theta1[:, 0, 2] = theta[:, 0]
                theta1[:, 1, 2] = theta[:, 1]
                theta1[:, 0, 0] = theta[:, 2]
                theta1[:, 1, 1] = theta[:, 3]
            elif mode == 'rotation_translation':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle)
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle)
                theta1[:, 0, 2] = theta[:, 1]
                theta1[:, 1, 2] = theta[:, 2]
            elif mode == 'rotation_translation_scale':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle) * theta[:, 3]
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle) * theta[:, 4]
                theta1[:, 0, 2] = theta[:, 1]
                theta1[:, 1, 2] = theta[:, 2]

        grid = F.affine_grid(theta1, torch.Size(x.shape), align_corners=False)
        img_transform = F.grid_sample(x, grid, padding_mode='border', align_corners=False)

        return img_transform, theta1