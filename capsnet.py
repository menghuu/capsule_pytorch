#!/usr/bin/env python3

"""
implement the capsnet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

BATCH_SIZE = 5
SHOULD_DOWNLOAD = False
DATASETS_PATH = './datas'
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATASETS_PATH, train=True, download=SHOULD_DOWNLOAD,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATASETS_PATH, train=False, download=False,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True
)


def squash(x):
    norm = torch.norm(x, 2, dim=-1, keepdim=True)
    norm_square = norm ** 2
    return (norm_square / (1 + norm_square)) * (x / norm)


def softmax(b):
    e = torch.exp(b)
    return e / torch.sum(e, dim=1, keepdim=True)


class CapsLayer(nn.Module):
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    bias = True

    def __init__(self, in_channels, out_channels, caps_size=8):
        super(CapsLayer, self).__init__()
        if caps_size in [-1, 0, 1]:
            self.caps_size = 1
        else:
            self.caps_size = caps_size
        self.conv2ds = nn.ModuleList([
            nn.Conv2d(
                in_channels, out_channels, self.kernel_size, stride=self.stride,
                padding=self.padding, dilation=self.dilation, groups=self.groups,
                bias=True)
            for _ in range(self.caps_size)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        # x is [batch_size, 256, 20, 20]
        xs = [conv.forward(x) for conv in self.conv2ds]
        # xs is [batch_size, 32, 6, 6, 8]
        xs = torch.stack(xs, -1)
        return self.relu(xs)
        # # [batch_size, 32*6*6, 8]
        # xs = xs.view(xs.size()[0], -1, self.caps_size)
        # # [batch_size, 32*6*6, 8]
        # #xs = self.squash(xs, 1)
        # return xs
    # def squash(self, x, dim):
    #     norm = torch.norm(x, 2, dim=dim, keepdim=True)
    #     norm_square = norm ** 2
    #     return (norm_square / (1 + norm_square)) * (x / norm)


class Route(nn.Module):
    l = 16
    h = 10

    def __init__(self, upper_conv_size=6, in_channels=32, caps_size=8,
                 fc=True, route_round=3):
        super(Route, self).__init__()
        # [32*6*6, 10, 16, 8]
        self.W = nn.Parameter(torch.randn(
            in_channels * upper_conv_size ** 2,
            self.h, self.l, caps_size
        ), requires_grad=True)
        self.route_round = route_round
        # [32*6*6]
        self.b = Variable(
            torch.zeros(
                in_channels * upper_conv_size ** 2
            ), requires_grad=False
        )

    def forward(self, x):
        # [batch_size, 32, 6, 6, 8]
        # [batch_size, 32*6*6, 8, 1]
        x = x.view(x.size()[0], -1, 8, 1)

        # x is [batch_size, 32*6*6, 10, 8, 1]
        x = torch.stack([x] * self.h, dim=2)
        # W is [32*6*6, 10, 16, 8]
        # W is [batch_size, 32*6*6, 10, 16, 8]
        extend_W = torch.stack([self.W] * x.size()[0], dim=0)

        # x is [batch_size, 32*6*6, 10, 16, 1]
        x = extend_W @ x
        # u is [batch_size, 32*6*6, 10, 16]
        u = torch.squeeze(x, dim=-1)
        # u is [batch_size, 32*6*6, 10, 16]
        # u is [batch_size, 10, 16, 32*6*6]
        u = u.transpose(1, 2).transpose(2, 3)
        for _ in range(self.route_round):
            # b is [32*6*6]
            # c is [32*6*6]
            c = self.softmax(self.b, dim=0)

            # [batch_size, 10, 16]
            s = u @ c
            # [batch_size, 10, 16]  ?????
            v = self.squash(s, 1)

            # [batch_size, 10, 16, 1]
            v = v.unsqueeze(dim=-1)

            shape = u.size(-1)
            self.b = self.b + \
                (u * v).view(-1, shape).mean(dim=0, keepdim=False)
            # [batch_size, 1, 1, 10, 16]
            # 1 个 channel
            # 长度为1
            # 高度为10
            # 每个点都是16的vector
            # [batch_size, 1, 1, 10, 16]
        return v.squeeze(dim=-1).unsqueeze(dim=1).unsqueeze(dim=1)

    def squash(self, x, dim):
        norm = torch.norm(x, 2, dim=dim, keepdim=True)
        norm_square = norm ** 2
        return (norm_square / (1 + norm_square)) * (x / norm)

    def softmax(self, b, dim):
        e = torch.exp(b)
        return e / torch.sum(e, dim=dim, keepdim=True)


class CapsNet(nn.Module):
    def __init__(self, in_channels=1):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 9)
        self.relu = nn.ReLU()
        self.primary_caps = CapsLayer(256, 32)
        self.digit_caps = Route()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        u = self.primary_caps(x)
        v = self.digit_caps(u)
        return v

    def margin_loss(self, predict, one_hot_labels):
        m_plus = 0.9
        m_minus = 0.1
        lambda_ = 0.5
        # one_hot_labels is [batch_size, 10]
        # predict is  [batch_size, 1, 1, 10, 16]
        # predict is  [batch_size, 10, 16]
        predict = predict.squeeze(dim=1).squeeze(dim=1)
        # [batch_size, 10]
        norm = predict.norm(p=2, dim=2, keepdim=False)
        zeros = torch.zeros_like(norm)
        lc = Variable(one_hot_labels, requires_grad=False) *\
            torch.max(zeros, m_plus - norm) ** 2 +\
            lambda_ * Variable(1 - one_hot_labels, requires_grad=False) * \
            torch.max(zeros, norm - m_minus) ** 2
        return lc.sum(dim=1, keepdim=False).mean(dim=0, keepdim=False)


def to_one_hot(x, length=10):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(BATCH_SIZE, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot


if __name__ == "__main__":
    model = CapsNet()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        target = to_one_hot(target)
        predict = model(Variable(data, requires_grad=False))
        loss = model.margin_loss(predict, target)
        loss.backward(retain_graph=True)
        optimizer.step()

        if idx % 1 == 0 and idx != 0:
            print("{:<3}: {}".format(idx,loss.data.tolist()[0]))
        # print('once')