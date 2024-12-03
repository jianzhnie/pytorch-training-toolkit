# License: BSD
# Author: Sasank Chilamkurthy

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner
from torch.optim import SGD
from torchvision import datasets, transforms


class MMResNet50(BaseModel):

    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        num_ftrs = self.resnet.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels


class Accuracy(BaseMetric):

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # 将一个批次的中间结果保存至 `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(accuracy=100 * total_correct / total_size)


def main() -> None:
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    data_dir = './vision_data/hymenoptera_data'
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    train_dataloader = torch.utils.data.DataLoader(image_datasets['train'],
                                                   batch_size=32,
                                                   shuffle=True,
                                                   num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(image_datasets['val'],
                                                 batch_size=32,
                                                 shuffle=True,
                                                 num_workers=4)

    runner = Runner(
        model=MMResNet50(),
        work_dir='./work_dir',
        train_dataloader=train_dataloader,
        # 优化器包装，用于模型优化，并提供 AMP、梯度累积等附加功能
        optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
        # 训练配置，例如 epoch 等
        train_cfg=dict(by_epoch=True, max_epochs=24, val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),
    )
    runner.train()


if __name__ == '__main__':
    main()
