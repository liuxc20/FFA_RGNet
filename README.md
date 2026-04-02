# FFA-RGNet
We developed an interpretable graph reasoning framework for enhanced diagnosis in fundus fluorescein angiography.

In this diagnostic study using multicenter FFA datasets, the proposed AI model outperformed nine baseline models and human experts at different expertise levels. The model also demonstrated a superior ability to recognize fine-grained semantic entities and learn disease-specific graph reasoning patterns aligned with clinical knowledge, while improving human experts’ diagnostic accuracy when used as an assistive tool.

This is the pytorch implementation for our paper.

## Requirements

- `torch>=1.6.0`
- `torchvision>=0.8.0`

## Datasets
We use three datasets in the paper: the Second Affiliated Hospital Zhejiang University School of Medicine (ZJU2, internal dataset), Taizhou First People’s Hospital and the Second Affiliated Hospital of Xi’an Jiaotong University (TZ and XJU2, external datasets). 

Data will be made available for research purposes upon request. 

## Run on ZJU2 dataset

- Run `bash train.sh` to train our model on the ZJU2 dataset.
  
- Run `bash test.sh` to test our model on the ZJU2 dataset.
