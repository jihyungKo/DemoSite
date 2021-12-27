import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from lib.loaders import medical_image_process as img_loader
from lib.loaders.medical_loader_utils import create_sub_volumes, get_viz_set

class CTORG(Dataset):
    """
    Code for reading the CT-ORG dataset
    """

    def __init__(self, mode, input_path, label_path, model_path, dataset_path='./datasets', classes=2, 
                crop_dim=(32, 32, 32), split_idx=90,
                 samples=100,
                 load=True):

        self.threshold = 0.0001
        self.normalization = False
        self.CLASSES = 2 
        self.crop_size = crop_dim
        self.full_vol_dim = (512, 512, 916)  # slice, width, height ###
        self.mode = mode 
        self.full_volume = None 
        self.affine = None ### changes it later but should be None in CovidSegmentation datasets
        self.list = []
        self.samples = samples #samples
        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])

        self.root = str(dataset_path)
        self.sub_vol_path = self.root + '/CT-ORG/CT_ORG_Data_Training/generated_6/' + mode + subvol + '/'

        self.training_path = self.root + '/CT-ORG/CT_ORG_Data_Training/'
        self.testing_path = self.root + '/CT-ORG/CT_ORG_Data_Testing/'
        self.save_name = self.root + '/CT-ORG/CT_ORG_Data_Training/ctorg-list-' + mode + '-samples-' + str(
            samples) + '.txt'

        if load:
            ## load pre-generated data
            # full_volume, affine
            
            #print(self.training_path)
            #list_IDs = sorted(glob.glob(os.path.join(self.training_path, '*.nii.gz')))
            #print("list_IDs : ", list_IDs)
            self.affine = img_loader.load_affine_matrix(input_path)
            
            #labels = sorted(glob.glob(os.path.join(self.root + '/CT-ORG/CT_ORG_Data_Training_labels/', '*.nii.gz')))
            #labels = labels[:split_idx]
            #list_IDs = list_IDs[:split_idx]
            self.full_volume = get_viz_set(input_path, label_path, dataset_name = "ctorg")
            return
        
    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        f, f_seg = self.list[index]
        img, img_seg = np.load(f), np.load(f_seg)
        # print("image.shpae:", img.shape)
        # print("imag_seg.shape:",img_seg.shape)

        # if self.mode == 'train' and self.augmentation:
        #     print(img.shape)
        #     print(img_seg.shape)

        #     img, img_seg = self.transform(img, img_seg)
        #     return torch.FloatTensor(img.copy()).unsqueeze(0), torch.FloatTensor(img_seg.copy())
        
        # print(img)
        return torch.FloatTensor(img).unsqueeze(0), torch.FloatTensor(img_seg)
        # return img, img_seg