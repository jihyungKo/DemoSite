import torch.optim as optim

from .Unet2D import Unet
from .Unet3D import UNet3D
from .Vnet import VNet, VNetLight

model_list = ['UNET3D', "UNET2D", 'VNET', 'VNET2']

def create_model(model_name):
    model_name = model_name
    assert model_name in model_list
    optimizer_name = 'sgd'
    lr = 1e-3
    in_channels = 1
    num_classes = 2
    weight_decay = 0.0000000001
    print("Building Model . . . . . . . ." + model_name)

    if model_name == 'VNET2':
        model = VNetLight(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name == 'VNET':
        model = VNet(in_channels=1, elu=False, classes=2)
    elif model_name == 'UNET3D':
        model = UNet3D(in_channels=in_channels, n_classes=num_classes)
    elif model_name == "UNET2D":
        model = Unet(in_channels, num_classes)

    print(model_name, 'Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
