from .lenet import *
from .vggnet import *
from .resnet import *
from .wide_resnet import *

def get_wide_resnet(model="28x10", pretrained=False):
    if model=="28x10":
        if pretrained:
            file_path = '/workspace/anomaly/wide-resnet.pytorch/checkpoint/pretrained/wide-resnet-28x10/wide-resnet-28x10.t7'
            checkpoint = torch.load(file_path)
            print("Load pretrained model")
            return checkpoint['net']
        else:
            return Wide_ResNet(28, 10, 0.3, 100)
    else:
        raise NotImplementedError