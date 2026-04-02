import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, args, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.split = split
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]

    def __len__(self):
        return len(self.examples)
        

class SequenceImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        exam_id = example['id']
        diagnosis = example['diagnosis_id']  
        image_infos = example['images']   

        images = []
        phases = []
        locations = []
        lesions = []

        for info in image_infos:
            img_path = os.path.join(self.image_dir, info['image_path'])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            images.append(image)  # Tensor
            phases.append(torch.tensor(info['phase'], dtype=torch.long))
            locations.append(torch.tensor(info['location'], dtype=torch.long))
            lesions.append(torch.tensor(info['lesion'], dtype=torch.float))  

        return {
            'exam_id': exam_id,
            'images': images,              
            'phases': phases,             
            'locations': locations,       
            'lesions': lesions,         
            'diagnosis': torch.tensor(diagnosis, dtype=torch.long)
        }

