import torch
print(torch.__version__)
torch.cuda.empty_cache()

import os
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from modules.dataloaders import ImageDataLoader
from modules.metrics import multi_class_evaluation,multi_label_metrics
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
import torch.hub
from model.model import FFA_RGNet

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,6,2'

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MultiTaskTransformer',help='the main model to be used.')

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/zju2/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/zju2/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='zju2', help='the dataset to be used.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=str, default='model/vit-base-patch16-224', help='whether to load the pretrained visual extractor')
    parser.add_argument('--pretrained_name', type=str, default='resnet',help='the pretrained model to be used.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=3, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='f1_score', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=25, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    
    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args


def compute_pos_weights(dataset, collate_fn):
    """
    lesion pos_weight for BCEWithLogitsLoss
    """
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    lesion_sum = torch.zeros(7)
    lesion_total = 0

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn  
    )

    for batch in tqdm(loader, desc="Computing lesion pos_weights"):
        lesions = batch['lesions'].squeeze(0) 
        lesion_sum += lesions.sum(dim=0)
        lesion_total += lesions.shape[0]

    pos_count = lesion_sum
    neg_count = lesion_total - lesion_sum
    pos_weight = neg_count / (pos_count + 1e-6)

    return pos_weight



        
def main():
    # parse arguments
    args = parse_agrs()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create data loader
    train_dataloader = ImageDataLoader(args,split='train', shuffle=True)
    val_dataloader = ImageDataLoader(args,split='val', shuffle=False)
    test_dataloader = ImageDataLoader(args,split='test', shuffle=False)
       

    
    # build model architecture
    model = FFA_RGNet(args)


    # compute model parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_params_million = total_params / 1000000
    print(f'Total parameters: {total_params_million:.2f} M')
    

    # compute pos_weight for lesion loss
    # pos_weight = compute_pos_weights(train_dataloader.dataset, train_dataloader.collate_fn)
    # print(pos_weight)
    pos_weight = torch.tensor([0.4257, 0.5097, 2.4929, 9.3322, 2.4005, 33.9016, 36.4051],  
                          dtype=torch.float).to(args.device)
 
    # get function handles of loss and metrics
    loss_fns = {
    'phase': nn.CrossEntropyLoss(),
    'location': nn.CrossEntropyLoss(),
    'lesion': nn.BCEWithLogitsLoss(pos_weight),
    'diagnosis': nn.CrossEntropyLoss()
}


    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    #set the metrics
    m_metrics = multi_class_evaluation
    ml_metrics = multi_label_metrics

    # build trainer and start to train (here test in every epoch)
    trainer = Trainer(model, loss_fns, m_metrics, ml_metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
