from data_modules.read_mod import ReadDataset
from data_modules.mask_mod import MaskFunc
from data_modules.trans_mod import DataTransform, DataTransform_Resnet
from data_modules.class_mod import ResNet50Module
from data_modules.actor_mod import PolicyModel
import torch
import fastmri.data.transforms as T
from torch.utils.data import DataLoader



def create_fastmri_dataset(args, partition):
    use_seed = (args.seed != 0) 
    if partition == 'train':  
                path=""
                LIST_PATH =''
    elif partition == 'val':            
                path=""
                LIST_PATH =''
    elif partition == 'test':
                path=""
                LIST_PATH = ''
    else:
        raise ValueError(f"partition should be in ['train', 'val', 'test'], not {partition}")


    if not args.train_resnet:
        dataset = ReadDataset(
                root=path,
                list_path=LIST_PATH, 
                data_partition=partition,
                args=args,
                transform=DataTransform(MaskFunc(args.center_fractions, args.accelerations), args.resolution, use_seed=use_seed, args=args),
                sample_rate=args.sample_rate,
            )
    else:
         dataset = ReadDataset(
                root=path,
                list_path=LIST_PATH, 
                data_partition=partition,
                args=args,
                transform=DataTransform_Resnet(mode=partition, use_seed=use_seed),
                sample_rate=args.sample_rate,
            )
    

    return dataset



def create_data_loader(args, partition, shuffle=False):
    dataset = create_fastmri_dataset(args, partition)

    if partition.lower() == 'train': 
        batch_size = args.batch_size 
    elif partition.lower() in ['val', 'test']:
        batch_size = args.val_batch_size 
    else:
        raise ValueError(f"'partition' should be in ('train', 'val', 'test'), not {partition}")
    
    loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last=True         
    )
    return loader


def load_varnet_model(batch, model, device):
    crop_size = batch.crop_size

    output1 = model(batch.masked_kspace.to(device), batch.mask.to(device).bool()).cpu() 

    if output1.shape[-1] < crop_size[1]:
        crop_size = (output1.shape[-1], output1.shape[-1])

    output = T.center_crop(output1, crop_size)
    return output, batch.slice_num, batch.fname

def load_infer_model(args, optim=False):
    checkpoint = torch.load(args.class_model_checkpoint, map_location=torch.device('cpu'), weights_only=False) 
    infer_args = checkpoint['hyper_parameters']
    infer_model = ResNet50Module(
        num_classes=infer_args.num_classes,
        dropout_prob=infer_args.dropout_prob
    )

    if not optim:
        for param in infer_model.parameters(): 
            param.requires_grad = False

    infer_model.load_state_dict(checkpoint['state_dict']) 
    final_infer_model = infer_model 
   
    return infer_args, final_infer_model

def load_policy_model(args):
    model = PolicyModel(
        resolution=args.resolution, 
        in_chans=1, 
        chans=args.num_chans, 
        num_pool_layers=args.num_layers, 
        drop_prob=args.drop_prob, 
        fc_size=args.fc_size, 
    )
    
    return model


def remove_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_policy_model_test(checkpoint_file, args,key):
    checkpoint = torch.load(checkpoint_file)
    model = load_policy_model(args)
    model_dict = remove_module_prefix(checkpoint[key]) 
    model.load_state_dict(model_dict)
    

    return model
