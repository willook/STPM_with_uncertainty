from torch import nn, overrides
from torchvision.models import resnet18


class ResnetVar(nn.Module):
    def __init__(self, norm_var=False,):
        super(ResnetVar, self).__init__()
        self.model_mean = resnet18(pretrained=False) # default: False
        self.model_var = resnet18(pretrained=False) # default: False
        self.norm_var = norm_var
    
    def forward(self, x):
        mean = self.model_mean(x)
        var = self.model_var(x)
        # TODO: var decay term이 있는데 normalize를 하는게 맞을까
        if self.norm_var:
            var_norm = var / var.max()
            var = var_norm
        return mean, var