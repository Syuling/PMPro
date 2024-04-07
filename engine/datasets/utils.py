import os
import torch
import torchvision
from torchvision.datasets.folder import default_loader
from engine.datasets import dataset_classes
from engine.tools.utils import load_json


def get_few_shot_setup_name(train_shot, seed):
    """Get the name for a few-shot setup.
    """
    return f"shot_{train_shot}-seed_{seed}"


class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor,eot_indices):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.eot_indices = eot_indices
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index]

    def __len__(self):
        return self.input_tensor.size(0)


class ITTensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, input_tensor2, input_tensor3, label_tensor):
        self.input_tensor = input_tensor
        self.input_tensor2 = input_tensor2
        self.input_tensor3 = input_tensor3
        self.label_tensor = label_tensor
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.input_tensor2[index], self.input_tensor3[index], self.label_tensor[index]
    
    def __len__(self):
        return self.input_tensor.size(0)

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index]
    
    def __len__(self):
        return self.input_tensor.size(0)

class TensorTwoDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor1, input_tensor2, label_tensor):
        self.input_tensor1 = input_tensor1
        self.input_tensor2 = input_tensor2
        self.label_tensor = label_tensor
    
    def __getitem__(self, index):
        return self.input_tensor1[index], self.input_tensor2[index], self.label_tensor[index]
    
    def __len__(self):
        return self.input_tensor1.size(0)
    
class TensorDatasetBlip(torch.utils.data.Dataset):
    def __init__(self, input_tensor1, input_tensor2,label_tensor):
        self.input_tensor1 = input_tensor1
        self.input_tensor2 = input_tensor2
        self.label_tensor = label_tensor
    
    def __getitem__(self, index):
        return self.input_tensor1[index], self.input_tensor2[index], self.label_tensor[index]
    
    def __len__(self):
        return self.input_tensor1.size(0)
    
class trainTensorDataset(torch.utils.data.Dataset):
    def __init__(self, img_tensor, txt_tensor,label_tensor):
        self.img_tensor = img_tensor
        self.txt_tensor = txt_tensor
        self.label_tensor = label_tensor
    
    def __getitem__(self, index):
        return self.img_tensor[index], self.txt_tensor[index], self.label_tensor[index]
    
    def __len__(self):
        return self.img_tensor.size(0)

class DatasetWrapperNew(torch.utils.data.Dataset):

    def __init__(self, data_source, transform):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        pl_img = default_loader(item['impath'])
        img = self.transform(pl_img)

        output = {
            "img": img,
            "pl_img":pl_img,
            "label": item['label'],
            "classname": item['classname'],
            "impath": item['impath'],
        }

        return output

class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, data_source, transform):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        img = self.transform(default_loader(item['impath']))

        output = {
            "img": img,
            "label": item['label'],
            "classname": item['classname'],
            "impath": item['impath'],
        }

        return output



def get_few_shot_benchmark(data_dir,
                           indices_dir,
                           dataset,
                           train_shot,
                           seed):
    # Check if the dataset is supported
    assert dataset in dataset_classes
    few_shot_index_file = os.path.join(
        indices_dir, dataset, f"{get_few_shot_setup_name(train_shot, seed)}.json")
    assert os.path.exists(few_shot_index_file), f"Few-shot data does not exist at {few_shot_index_file}."
    benchmark = dataset_classes[dataset](data_dir)
    few_shot_dataset = load_json(few_shot_index_file)
    return {
        'train': few_shot_dataset['train']['data'],
        'val': few_shot_dataset['val']['data'],
        'test': benchmark.test,
        'lab2cname': benchmark.lab2cname,
        'classnames': benchmark.classnames,
    }


def get_train_benchmark(data_dir,dataset):
    # Check if the dataset is supported
    assert dataset in dataset_classes
    benchmark = dataset_classes[dataset](data_dir)
    return benchmark.train


def get_testset(dataset, data_dir):
    if dataset in dataset_classes:
        benchmark = dataset_classes[dataset](data_dir)
        return benchmark.test
    else:
        raise NotImplementedError()


def get_label_map(data_dir, dataset_name):
    if dataset_name in ['imagenet_a', 'imagenet_r']:
        benchmark = dataset_classes[dataset_name](data_dir)
        return benchmark.label_map
    else:
        return None