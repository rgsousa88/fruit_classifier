## Introduction

In this branch, was compared the models performance using fixed hyperparameters (which can found at config.json). 

The architectures supported are:

* 'resnet50' and 'resnet18',
* 'mobilenet_v3' (large and small versions),
* 'efficientnet (B0, B1 and B2),
* 'vit_b_16' and 'vit_b_32',
* 'swin_t' and 'swin_b'
* 'swin_v2_t' and 'swin_v2_s'

For each architecture, it was used pretrained models provided by PyTorch (https://docs.pytorch.org/vision/main/models.html).

## Experiments ##

All architectures were trained for 30 epochs using the predefined training split from the Fruit Dataset.
The training configuration employed the Adam optimizer with an initial learning rate of 0.001, coupled with a step learning rate scheduler that reduces the rate after every 10 epochs.
Due to hardware constraints of the RTX 4070 laptop GPU, the batch size was set to 10.
This setup ensured stable training across all model architectures while efficiently utilizing the available computational resources.

For training Transformer based architecures, it was necessary to run training script in Lightinging.AI environment with a T4 GPU. In this case, a batch size of 100 was used.

## Results ##

The table below shows the best results (validation accuracy) obtained for each architecure after training.

| Architecture | Accuracy | Notes |
|-------------|----------|-------|
| **EfficientNet-B1** | **95.0%** | 🥇 Best performer |
| EfficientNet-B0 | 94.5% | 🥈 Close second |
| EfficientNet-B2 | 94.0% | |
| MobileNet-V3 Large | 94.0% | Tied with EfficientNet-B2 |
| MobileNet-V3 Small | 91.5% | Good efficiency |
| ResNet-50 | 89.5% | |
| ResNet-18 | 86.0% | |
| ViT-16 | 61.0% | ⚠️ Poor performance |
| ViT-32 | 48.0% | ⚠️ Poor performance |



