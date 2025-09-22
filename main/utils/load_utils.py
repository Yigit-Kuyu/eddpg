import os
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
                LIST_PATH =''
    else:
        raise ValueError(f"partition should be in ['train', 'val', 'test'], not {partition}")

    print(f"Creating dataset for {partition}:")
    print(f"  Data path: {path}")
    print(f"  Label path: {LIST_PATH}")
    print(f"  Path exists: {os.path.exists(path)}")
    print(f"  Label file exists: {os.path.exists(LIST_PATH)}")

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
    
    print(f"  Dataset created with {len(dataset)} samples")
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
        drop_last=False          
    )
    print(f"  DataLoader created: {len(loader)} batches with batch_size={batch_size}")
    return loader


def load_varnet_model(batch, model, device):
    crop_size = batch.crop_size

    output1 = model(batch.masked_kspace.to(device), batch.mask.to(device).bool()).cpu() 

    if output1.shape[-1] < crop_size[1]:
        crop_size = (output1.shape[-1], output1.shape[-1])

    output = T.center_crop(output1, crop_size)
    return output, batch.slice_num, batch.fname


def load_infer_model(args, optim=False): 
    
    ckpt = torch.load(args.class_model_checkpoint, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)

    def strip_prefix(d, pfx):
        return { (k[len(pfx):] if k.startswith(pfx) else k): v for k, v in d.items() }
    sd = strip_prefix(sd, 'module.')
    sd = strip_prefix(sd, 'model.')

    infer_args  = ckpt.get('hyper_parameters', None)  #
    infer_model =  ResNet50Module( 
        num_classes=infer_args.num_classes,
        dropout_prob=infer_args.dropout_prob
    ) 

    return infer_args, infer_model 


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


def load_resnet_model_for_inference(args):
    
    if args.class_model_checkpoint.endswith('.pth'):
        model_path = args.class_model_checkpoint
    else:
        model_path = os.path.join(args.class_model_checkpoint, 'best_model.pth')
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    sd = checkpoint.get('state_dict', checkpoint)
    
    def strip_prefix(d, pfx):
        return {(k[len(pfx):] if k.startswith(pfx) else k): v for k, v in d.items()}
    sd = strip_prefix(sd, 'module.')
    sd = strip_prefix(sd, 'model.')
    
    return sd


def load_policy_model_test(checkpoint_file, args,key):
    checkpoint = torch.load(checkpoint_file)
    model = load_policy_model(args)
    model_dict = remove_module_prefix(checkpoint[key]) 
    model.load_state_dict(model_dict)
    

    return model
