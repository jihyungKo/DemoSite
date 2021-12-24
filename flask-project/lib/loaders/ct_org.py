import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import lib.augment3D as augment3D
import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import create_sub_volumes, get_viz_set


class CTORG(Dataset):
    """
    Code for reading the CT-ORG dataset
    """

    def __init__(self, args, mode, dataset_path='./datasets', classes=2, 
                crop_dim=(32, 32, 32), split_idx=90,
                 samples=100,
                 load=True):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param split_idx: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.threshold = args.threshold
        self.normalization = args.normalization
        
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
            print(self.training_path)
            list_IDs = sorted(glob.glob(os.path.join(self.training_path, '*.nii.gz')))
            print(len(list_IDs))
            self.affine = img_loader.load_affine_matrix(list_IDs[0])
            self.list = utils.load_list(self.save_name)  # EOFError: Ran out of input
            labels = sorted(glob.glob(os.path.join(self.root + '/CT-ORG/CT_ORG_Data_Training_labels/', '*.nii.gz')))
            if self.mode == 'train':
                labels = labels[:split_idx]
                list_IDs = list_IDs[:split_idx]
            else:
                labels = labels[split_idx:]
                list_IDs = list_IDs[split_idx:]
            self.full_volume = get_viz_set(list_IDs, labels, dataset_name = "ctorg")
            #self.full_volume = self.list[0]
            #f, f_seg = self.list[0]
            #img, img_seg = np.load(f), np.load(f_seg)
            #temp = torch.stack([torch.FloatTensor(img), torch.FloatTensor(img_seg)], dim = 0)
            #self.full_volume = temp
            return
        
       
        utils.make_dirs(self.sub_vol_path)
        
        self.train_images, self.train_labels, self.val_labels, self.val_images = [], [], [], []
        list_images = sorted(
            glob.glob(os.path.join(dataset_path, 'CT-ORG/CT_ORG_Data_Training/', '*.nii.gz')))

        list_labels = sorted(glob.glob(os.path.join(dataset_path, 'CT-ORG/CT_ORG_Data_Training_labels/', '*.nii.gz')))
        len_of_data = len(list_images)

        self.affine = img_loader.load_affine_matrix(list_images[0]) # nii.gz file format
        
        #self.train_images.append(list_images[0])
        #self.train_labels.append(list_labels[0])

        self.val_images.append(list_images[1])
        self.val_labels.append(list_labels[1])
        '''
        for i in range(len_of_data):
            if i >= 0 and i < split_idx:
                self.train_images.append(list_images[i])
                self.train_labels.append(list_labels[i])
            else:
                self.val_images.append(list_images[i])
                self.val_labels.append(list_labels[i])
        '''
        if (mode == 'train'):
            self.list_IDs = list_images
            self.list_labels = list_labels

        elif (mode == 'val'):
            self.list_IDs = list_images[split_idx:]
            self.list_labels = list_labels[split_idx:]
        
        
        self.list = create_sub_volumes(self.list_IDs, self.list_labels, dataset_name="ctorg", mode=mode,
                                       samples=samples, full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                       sub_vol_path=self.sub_vol_path, th_percent=self.threshold,
                                       normalization=self.normalization)
        print("{} SAMPLES =  {}".format(mode, len(self.list)))
        
        
        
        
        utils.save_list(self.save_name, self.list)
        # normalization=self.normalization,th_percent=self.threshold
        '''
        self.normalization = args.normalization
        #self.augmentation = 'full_volume_mean' #args.augmentation
        self.augmentation = args.augmentation
        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                            augment3D.ElasticTransform()], p=0.5)

        self.classes = classes


        
        

        
        
        if load:
            ## load pre-generated data
            print(self.training_path)
            list_IDs = sorted(glob.glob(os.path.join(self.training_path, '*.nii.gz')))
            print(len(list_IDs))
            self.affine = img_loader.load_affine_matrix(list_IDs[0])
            self.list = utils.load_list(self.save_name)  # EOFError: Ran out of input
            labels = sorted(glob.glob(os.path.join(self.root + '/CT-ORG/CT_ORG_Data_Training_labels/', '*.nii.gz')))
            if self.mode == 'train':
                labels = labels[:split_idx]
                list_IDs = list_IDs[:split_idx]
            else:
                labels = labels[split_idx:]
                list_IDs = list_IDs[split_idx:]
            self.full_volume = get_viz_set(list_IDs, labels, dataset_name = "ctorg")
            return

        list_IDs = sorted(glob.glob(os.path.join(self.training_path, '*.nii.gz')))
        labels = sorted(glob.glob(os.path.join(self.root + '/CT-ORG/CT_ORG_Data_Training_labels/', '*.nii.gz')))

        self.affine = img_loader.load_affine_matrix(list_IDs[0]) # nii.gz file format
        print(len(list_IDs))
        
        
        if self.mode == 'train':
            list_IDs = list_IDs[:split_idx]
            labels = labels[:split_idx]

            self.list = create_sub_volumes(list_IDs, labels,
                                           dataset_name="CT-ORG", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, normalization=self.normalization,
                                           th_percent=self.threshold)
        elif self.mode == 'val':
            list_IDs = list_IDs[split_idx:]
            labels = labels[split_idx:]
            self.list = create_sub_volumes(list_IDs, labels,
                                           dataset_name="CT-ORG", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, normalization=self.normalization,
                                           th_percent=self.threshold)

        elif self.mode == 'test':
            self.list_IDs = sorted(glob.glob(os.path.join(self.testing_path, '*.nii.gz')))
            self.labels = None
        '''
        

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