from torch.utils.data import Dataset
from math import floor
from pathlib import Path
from typing import NamedTuple, Dict, Any
import random
from collections import Counter
import torch
import h5py
import pandas as pd
import numpy as np
import xml.etree.ElementTree as etree
import os

def et_query(root, qlist, namespace="http://www.ismrm.org/ISMRMRD"):
    s = "."
    prefix = "ismrmrd_namespace"
    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    label: int
    metadata: Dict[str, Any]

class ReadDataset(Dataset):
    def __init__(self, root, list_path, data_partition, args, transform=None, sample_rate=None):
    
        self.transform = transform  
        self.data_partition = data_partition
        self.raw_samples = []
        self.ce_weights = None 
        label_list = self.read_sample_label(list_path)
        files = list(Path(root).iterdir())
        for fname in sorted(files):
            metadata, num_slices,_ = self._retrieve_metadata(fname)
            new_raw_samples = []
            if sample_rate < 1.0: 
                    half_slice = 0.5 * num_slices
                    start = floor(half_slice - 0.5 * sample_rate * num_slices)
                    end = floor(half_slice + 0.5 * sample_rate * num_slices)
                    for slice_ind in range(start, end):
                        label = self.find_label(label_list, self.remove_h5_extension(fname), slice_ind)
                        raw_sample = FastMRIRawDataSample(fname, slice_ind, label, metadata)
                        new_raw_samples.append(raw_sample)
            else: 
                    for slice_ind in range(num_slices):
                        label = self.find_label(label_list, self.remove_h5_extension(fname), slice_ind)
                        raw_sample = FastMRIRawDataSample(fname, slice_ind, label, metadata)
                        new_raw_samples.append(raw_sample)
            self.raw_samples += new_raw_samples

        label_distribution = self.count_label_distribution()
        if args.train_resnet: self.raw_samples +=self.oversample_minority(label_distribution)  
        counts = Counter(sample.label for sample in self.raw_samples)  
        total  = sum(counts.values())
        self.ce_weights= torch.tensor([total/counts[0], total/counts[1]],dtype=torch.float) 
        if data_partition == 'train': random.shuffle(self.raw_samples)
    
    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i):
        fname, dataslice, label, metadata = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice] 
            mask = np.asarray(hf["mask"]) if "mask" in hf else None 
            target = hf["reconstruction_rss"][dataslice] if "reconstruction_rss"  in hf else None  
            attrs = dict(hf.attrs)
            attrs.update(metadata)
        
        sample=self.transform(kspace, mask, target, attrs, fname.name, dataslice, label) 
        meta = sample[-1]
        meta['ce_weights'] = self.ce_weights
        return sample

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, 'r') as hf:
            et_root = etree.fromstring(hf['ismrmrd_header'][()])
            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            ) #  (640, 368, 1)
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            ) 

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"])) 
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1 

            padding_left = enc_size[1] // 2 - enc_limits_center 
            padding_right = padding_left + enc_limits_max 
            num_slices = hf["kspace"].shape[0] 
            image_size = [hf["kspace"].shape[2], hf["kspace"].shape[3]] 
            
            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs
            }

        return metadata, num_slices, image_size

    def read_sample_label(self, list_path):
        label_df = pd.read_csv(list_path, header=0, names=['file', 'slice', 'label'])
        return label_df

    def find_label(self, label_list, target_fname, target_slice):
        filtered_rows = label_list.loc[(label_list['file'] == target_fname) & (label_list['slice'] == target_slice)]
        if not filtered_rows.empty:
            return int(filtered_rows['label'].values[0])
        else:
            return int(0) 

    def remove_h5_extension(self, fname):
        return os.path.splitext(fname.name)[0]

    def count_label_distribution(self):
        labels = [sample.label for sample in self.raw_samples]
        label_distribution = Counter(labels)
        return label_distribution

    def oversample_minority(self, label_dist):
        oversampled_raw_samples = []
        if self.data_partition == 'train':
            max_samples = max(label_dist.values())
            for label, count in label_dist.items():
                oversample_factor = max_samples // count
                if oversample_factor > 1:
                    minority_samples = [sample for sample in self.raw_samples if sample.label == label]
                    oversampled_raw_samples.extend(minority_samples * (oversample_factor - 1))
        return oversampled_raw_samples
