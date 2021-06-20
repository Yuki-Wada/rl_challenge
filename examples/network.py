import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, padding=pad)

    def __call__(self, x):
        h = self.conv(x)
        return F.relu(h)

class SLPolicy(nn.Module):
    '''
    Supervised learning policy
    '''
    def __init__(self):
        in_channels = 2
        ksize = 3
        super(SLPolicy, self).__init__()

        self.block1 = Block(in_channels, 64, ksize)
        self.block2 = Block(64, 128, ksize)
        self.block3 = Block(128, 128, ksize)
        self.block4 = Block(128, 128, ksize)
        self.block5 = Block(128, 128, ksize)
        self.block6 = Block(128, 128, ksize)
        self.block7 = Block(128, 128, ksize)
        self.block8 = Block(128, 128, ksize)
        self.conv9 = nn.Conv2d(128, 1, 1, bias=False)
        self.register_parameter('bias10', nn.Parameter(torch.zeros(64), requires_grad=True))

    def __call__(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.block8(h)
        h = self.conv9(h)
        h = h.reshape((-1, 64))
        h = h + self.bias10
        h = nn.Softmax(dim=1)(h)
        return h

class RolloutPolicy(nn.Module):
    '''
    Rollout policy: Works faster but worse than SL policy
    '''
    def __init__(self):
        super(RolloutPolicy, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, bias=False, padding=1)
        self.register_parameter('bias2', nn.Parameter(torch.zeros(64), requires_grad=True))

    def __call__(self, x):
        h = self.conv1(x)
        h = h.view((-1,64))
        h = h + self.bias2
        h = nn.Softmax(dim=1)(h)
        return h

class Value(nn.Module):
    def __init__(self):
        in_channels = 2
        ksize = 3
        super(Value, self).__init__()

        self.block1 = Block(in_channels, 64, ksize)
        self.block2 = Block(64, 128, ksize)
        self.block3 = Block(128, 128, ksize)
        self.block4 = Block(128, 128, ksize)
        self.block5 = Block(128, 128, ksize)
        self.block6 = Block(128, 128, ksize)
        self.block7 = Block(128, 128, ksize)
        self.block8 = Block(128, 128, ksize)
        self.block9 = Block(128, 1, ksize)
        self.fc10 = nn.Linear(64, 128, bias=False)
        self.fc11 = nn.Linear(128, 1, bias=False)

    def __call__(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.block8(h)
        h = self.block9(h)
        h = h.view((-1, 64))
        h = self.fc10(h)
        h = F.dropout(h, 0.4)
        h = self.fc11(h).view(-1)
        return h
