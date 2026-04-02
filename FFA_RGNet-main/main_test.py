import torch
torch.cuda.empty_cache()
print(torch.__version__)
import torch.nn as nn
import argparse
import numpy as np
from modules.dataloaders import ImageDataLoader
from modules.metrics import multi_class_evaluation
from modules.tester import Tester
from model.model import FFA_RGNet

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MultiTaskTransformer',help='the main model to be used.')

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='zju2', help='the dataset to be used.')
    parser.add_argument('--image_mode', type=str, default='single', help='single or multi.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=str, default=None, help='whether to load the pretrained visual extractor')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=2, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='f1_score', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')
    # Learning Rate Scheduler
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--load', type=str, help='whether to load a pre-trained model.')

    args = parser.parse_args()
    return args

        
def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create data loader
    test_dataloader = ImageDataLoader(args,split='test', shuffle=False)

    
    # build model architecture 
    model = FFA_RGNet(args)

    loss_fns = {
    'phase': nn.CrossEntropyLoss(),
    'location': nn.CrossEntropyLoss(),
    'lesion': nn.BCEWithLogitsLoss(),
    'diagnosis': nn.CrossEntropyLoss()
}

    multi_class_metrics = multi_class_evaluation

    # build tester and start to test (here test in every epoch)
    tester = Tester(model, loss_fns,multi_class_metrics, args, test_dataloader)
    tester.test()
    

if __name__ == '__main__':
    main()