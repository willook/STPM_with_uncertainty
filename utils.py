import torch

# turn on training option dropout layer
def turn_on_dropout(module: torch.nn.Module):
    is_leaf = True
    for sub_module in module.children():
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train
        is_leaf = False
        turn_on_dropout(sub_module)
    
    if is_leaf and isinstance(module, torch.nn.Dropout):
        module.train(True)

def print_modules(module: torch.nn.Module):
    is_leaf = True
    #print(module)
    for sub_module in module.children():
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train
        is_leaf = False
        print_modules(sub_module)
    
    if is_leaf:
        print(module.training, module)
        #module.train(True)

