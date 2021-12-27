import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import lib.loaders as medical_loaders
import lib.models as models
from lib.visual import visualize_3D_no_overlap_new
from lib.criterion import DiceLoss

from lib.loaders import medical_image_process as img_loader

def execute(input_path, label_path, model_path, output_path):
    print("inference 접근")
    seed = 1777777
    torch.manual_seed(seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    dataset_name = "ctorg"
    full_volume, affine = medical_loaders.select_full_volume_for_infer(dataset_name, input_path, label_path, model_path)
    # VNET, UNET3D, UNET2D
    model_name = "VNET"
    model, optimizer = models.create_model(model_name)
    criterion = DiceLoss(classes=2)
    
    # ## TODO LOAD PRETRAINED MODEL
    print("affine   :   ", affine.shape)
    # no affine
    pretrained = model_path0
    model.restore_checkpoint(pretrained)
   
    '''
    if args.cuda:
        model = model.cuda()
        full_volume = full_volume.cuda()
        print("Model transferred in GPU.....")
    '''
    
    # print("full   :   ", full_volume.shape)
    # a,b,c,d = full_volume.shape
    dim = (32,32,32)
    visualize_3D_no_overlap_new(output_path, model_name, dataset_name, full_volume, affine, model, 10, dim)
    
'''
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loadData', default=True)
    parser.add_argument('--threshold', default=0.0001)
    parser.add_argument('--normalization', default=False)
    parser.add_argument('--augmentation', default=False)

    parser.add_argument('--batchSz', type=int, default=1)
    
    parser.add_argument('--dataset_name', type=str, default="ctorg")
    parser.add_argument('--dim', nargs="+", type=int, default=(512, 512, 512))
    parser.add_argument('--nEpochs', type=int, default=100)

    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--samples_train', type=int, default=90)
    parser.add_argument('--samples_val', type=int, default=10)
    parser.add_argument('--split', type=float, default=0.9)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='VNET',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--pretrained',
                        default='../saved_models/VNET_checkpoints/VNET_04_11___18_34_ctorg_/VNET_04_11___18_34_ctorg__last_epoch.pth',
                        type=str, metavar='PATH',
                        help='path to pretrained model')

    args = parser.parse_args()

    import os
    #os.makedirs('../inference_checkpoints/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(utils.datestr(), args.dataset_name))
    # args.save = 
    #args.tb_log_dir = '../runs/'
    return args
'''