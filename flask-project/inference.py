import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Lib files
import lib.utils as utils
import lib.medloaders as medical_loaders
import lib.models as models
from lib.visual import visualize_3D_no_overlap_new
from lib.criterion import DiceLoss

from lib.loaders import medical_image_process as img_loader
#

def main():
    args = get_arguments()
    seed = 1777777
    utils.reproducibility(args, seed)

    full_volume, affine = medical_loaders.select_full_volume_for_infer(args,
                                                                        path='./datasets')
    model, optimizer = models.create_model(args)
    criterion = DiceLoss(classes=args.classes)
    
    # ## TODO LOAD PRETRAINED MODEL
    print("affine   :   ", affine.shape)
    # no affine
    model.restore_checkpoint(args.pretrained)
    
    if args.cuda:
        model = model.cuda()
        full_volume = full_volume.cuda()
        print("Model transferred in GPU.....")
    
    print("full   :   ", full_volume.shape)

    a,b,c,d = full_volume.shape
    visualize_3D_no_overlap_new(args, full_volume, affine, model, 10, args.dim)

    ## TODO TARGET FOR LOSS

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
    #args.save = '../inference_checkpoints/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(utils.datestr(), args.dataset_name)
    #args.tb_log_dir = '../runs/'
    return args

if __name__ == '__main__':
    main()