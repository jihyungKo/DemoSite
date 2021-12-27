from torch.utils.data import DataLoader
from torch.utils.data.dataset import T

from .ct_org import CTORG

def generate_datasets(args, path='./datasets'):
    params = {'batch_size': args.batchSz,
              'shuffle': False,
              'num_workers': 2}
    samples_train = args.samples_train
    samples_val = args.samples_val
    split_percent = args.split

    if args.dataset_name == "ctorg":
        train_loader = CTORG(args, 'train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                              split_idx=samples_train, samples=samples_train, load=args.loadData) #args.loadData)

        val_loader = CTORG(args, 'val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                            split_idx=samples_train,
                                            samples=samples_val, load=args.loadData) #args.loadData)
    print("train_loader", len(train_loader)) 
    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator, val_loader.full_volume, val_loader.affine


def select_full_volume_for_infer(dataset_name, input_path, label_path, model_path):
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}

    samples_train = 90
    samples_val = 10
    split_percent = 0.9
    classes = 2
    dim = (32,32,32)
    loadData = True

    if dataset_name == "ctorg":
        loader = CTORG('train', input_path, label_path, model_path, classes=classes, crop_dim=dim,
                                              split_idx=samples_train, samples=samples_train, load=loadData) #args.loadData)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return loader.full_volume, loader.affine