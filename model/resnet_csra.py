"""
Class-specific residual attention (CSRA) module.

Adapted from:
    Zhu, K., & Wu, J. (2021). "Residual Attention: A Simple but Effective Method
    for Multi-Label Recognition." ICCV 2021, pp. 184-193.
    arXiv:2108.02456 - https://arxiv.org/abs/2108.02456
    Reference implementation: https://github.com/Kevinz-code/CSRA
"""

from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
from .csra import CSRA, MHA
import torch.utils.model_zoo as model_zoo
import logging
import torch
import torch.nn as nn
import os

from loss.asl_loss import AsymmetricLossOptimized
from loss.caal import CAAL

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet_CSRA(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_heads, lam, num_classes, depth=101, input_dim=2048, cutmix=None):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet_CSRA, self).__init__(self.block, self.layers)
        self.init_weights(pretrained=True, cutmix=cutmix)

        self.classifier = MHA(num_heads, lam, input_dim, num_classes).to(device)
        # self.loss_func = CAAL(10, 8.0, 2.0, 12.0, 0.8, 0.2)
        self.loss_func = AsymmetricLossOptimized(gamma_neg=5, gamma_pos=3, clip=0.1, disable_torch_grad_focal_loss=True).to(device)
        # self.loss_func = nn.BCELoss()

    def backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def forward_train(self, x, target):
        x = self.backbone(x)
        logit = nn.Sigmoid()(self.classifier(x))
        loss = self.loss_func(logit, target)
        return logit, loss

    def forward_test(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def forward(self, x, target=None):
        if target is not None:
            return self.forward_train(x, target)
        else:
            return self.forward_test(x)

    def init_weights(self, pretrained=True, cutmix=None):
        if cutmix is not None:
            print("backbone params inited by CutMix pretrained model")
            state_dict = torch.load(cutmix)
        elif pretrained:
            print("backbone params inited by Pytorch official model")
            model_url = model_urls["resnet{}".format(self.depth)]
            state_dict = model_zoo.load_url(model_url)

        model_dict = self.state_dict()
        try:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            self.load_state_dict(pretrained_dict)
        except:
            logger = logging.getLogger()
            logger.info(
                "the keys in pretrained model is not equal to the keys in the ResNet you choose, trying to fix...")
            state_dict = self._keysFix(model_dict, state_dict)
            self.load_state_dict(state_dict)

        # remove the original 1000-class fc
        self.fc = nn.Sequential() 
