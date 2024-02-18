import os
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path
from torchvision import models
import torch
import torch.nn as nn
from torchsummary import summary
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = models.vgg16(pretrained=self.config.params_pretrained)
        self.model = self.model.to(self.config.params_device)
        return self.model
    
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for param in model.parameters()[:-freeze_till]:
                param.requires_grad = False

        n_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, classes), nn.LogSoftmax(dim=1))
        # multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        return model
    
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None
        )
        self.full_model = self.full_model.to(self.config.params_device)
        optimizer_top = torch.optim.Adam(self.full_model.parameters(), lr=self.config.params_learning_rate)
        checkpoint = {'model_state_dict': self.full_model.state_dict(),
                        'optimizer_state_dict': optimizer_top.state_dict()}
        summary(self.full_model, input_size=tuple(self.config.params_image_size), batch_size=self.config.params_batch_size, device=self.config.params_device)
        self.save_model(checkpoint=checkpoint, path=self.config.updated_base_model_path)

    
    @staticmethod
    def save_model(path: Path, checkpoint: dict):
        torch.save(checkpoint, path)