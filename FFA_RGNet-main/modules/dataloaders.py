import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import SequenceImageDataset  

class ImageDataLoader(DataLoader):
    def __init__(self, args, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        self.dataset = SequenceImageDataset(self.args, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'pin_memory': True,
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(batch):
        """
        batch: list of dict, each dict includes:
            'exam_id': str,
            'images': list of tensor (C,H,W),
            'phases': list of tensor (scalar),
            'locations': list of tensor (scalar),
            'lesions': list of tensor (multi-hot vector),
            'diagnosis': tensor(scalar)
        """
        batch_size = len(batch)
        max_len = max(len(sample['images']) for sample in batch)

        C, H, W = batch[0]['images'][0].shape
        lesion_dim = batch[0]['lesions'][0].shape[0]

        # 预分配张量
        images_tensor = torch.zeros((batch_size, max_len, C, H, W), dtype=torch.float)
        mask_tensor = torch.zeros((batch_size, max_len), dtype=torch.bool)
        phases_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
        locations_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
        lesions_tensor = torch.zeros((batch_size, max_len, lesion_dim), dtype=torch.float)
        diagnosis_tensor = torch.zeros((batch_size,), dtype=torch.long)
        exam_ids = []

        for i, sample in enumerate(batch):
            seq_len = len(sample['images'])
            exam_ids.append(sample['exam_id'])
            images_tensor[i, :seq_len] = torch.stack(sample['images'])
            phases_tensor[i, :seq_len] = torch.stack(sample['phases'])
            locations_tensor[i, :seq_len] = torch.stack(sample['locations'])
            lesions_tensor[i, :seq_len] = torch.stack(sample['lesions'])
            mask_tensor[i, :seq_len] = 1
            diagnosis_tensor[i] = sample['diagnosis']

        return {
            'exam_ids': exam_ids,
            'images': images_tensor,        # (B, T_max, C, H, W)
            'mask': mask_tensor,            # (B, T_max)
            'phases': phases_tensor,        # (B, T_max)
            'locations': locations_tensor,  # (B, T_max)
            'lesions': lesions_tensor,      # (B, T_max, L)
            'diagnosis': diagnosis_tensor   # (B,)
        }
