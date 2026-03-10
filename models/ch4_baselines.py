import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from nets.light_cnn_sim import LightCNN_29Layers_v2


class SubCenterArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, k=2, s=16.0, m=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, k, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding, label=None, apply_margin=True):
        embedding = F.normalize(embedding, p=2, dim=1)
        weight = F.normalize(self.weight.view(-1, self.in_features), p=2, dim=1)
        cosine_all = F.linear(embedding, weight).view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)

        if not apply_margin or label is None:
            return cosine * self.s

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.s


def _build_resnet50(pretrained=False):
    try:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
    except AttributeError:
        backbone = models.resnet50(pretrained=pretrained)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, in_features


def _build_densenet121(pretrained=False):
    try:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)
    except AttributeError:
        backbone = models.densenet121(pretrained=pretrained)
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Identity()
    return backbone, in_features


class ArcFaceClosedSetModel(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_name,
        embedding_dim=512,
        sub_centers_k=2,
        arcface_s=16.0,
        arcface_m=0.2,
        pretrained=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone_name
        self.fd = None

        if backbone_name == "lightcnn":
            self.backbone = LightCNN_29Layers_v2(num_classes=num_classes)
            backbone_dim = 512
        elif backbone_name == "resnet50":
            self.backbone, backbone_dim = _build_resnet50(pretrained=pretrained)
        elif backbone_name == "densenet121":
            self.backbone, backbone_dim = _build_densenet121(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if backbone_dim == embedding_dim:
            self.embedding_head = nn.Identity()
        else:
            self.embedding_head = nn.Sequential(
                nn.Linear(backbone_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
            )

        self.arcface_head = SubCenterArcFaceHead(
            in_features=embedding_dim,
            out_features=num_classes,
            k=sub_centers_k,
            s=arcface_s,
            m=arcface_m,
        )

    def _extract_features(self, image):
        image = (image + 1.0) / 2.0
        features = self.backbone(image)
        embedding = self.embedding_head(features)
        self.fd = F.normalize(embedding, p=2, dim=1)
        return self.fd

    def forward(self, img1, img2=None, subject_mask=None, label=None, mode="train"):
        embedding = self._extract_features(img1)
        if mode == "predict":
            return embedding
        if mode == "eval_logits":
            return self.arcface_head(embedding, label=None, apply_margin=False)
        return self.arcface_head(embedding, label=label, apply_margin=True)

    def compute_center_loss(self, label):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def compute_auxiliary_loss(self, label):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def compute_reconstruction_loss(self, orig_img):
        return torch.tensor(0.0, device=next(self.parameters()).device)


class LightCNNSubCenterArcFace(ArcFaceClosedSetModel):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, backbone_name="lightcnn", **kwargs)


class ResNet50SubCenterArcFace(ArcFaceClosedSetModel):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, backbone_name="resnet50", **kwargs)


class DenseNet121SubCenterArcFace(ArcFaceClosedSetModel):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, backbone_name="densenet121", **kwargs)
