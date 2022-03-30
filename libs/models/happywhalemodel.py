from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import externals as ex

class BaseEffnetModel(nn.Module):
    def __init__(
        self, model_name, embedding_size, num_classes, 
        s, m, easy_margin, ls_eps, pretrained=True, freeze_backbone=False
    ):
        super(BaseEffnetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = ex.GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = ex.ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=s,
            m=m,
            easy_margin=easy_margin,
            ls_eps=ls_eps
        )

    def forward(self, images, labels):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        output = self.fc(embedding, labels)
        return output
    
    def extract(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        return embedding
