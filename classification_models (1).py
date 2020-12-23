import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn
from aux import load_weights


class MapNet(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MapNet, self).__init__()
        self.in_features = feature_dim

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        self.classification = nn.Linear(in_features=self.in_features, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.mapping(x))
        x = self.classification(x)
        return x

class MapNet_dropout(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MapNet_dropout, self).__init__()
        self.in_features = feature_dim

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        self.classification = nn.Linear(in_features=self.in_features, out_features=num_classes)
        
    def forward(self, x):
        x = F.relu(self.mapping(x))
        x = self.classification(x)
        return x


class MapNet_bigger(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MapNet_bigger, self).__init__()
        self.in_features = feature_dim

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        self.classification = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=256),
            nn.ReLU(inplace=False), 
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=False), 
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x):
        x = F.relu(self.mapping(x))
        x = self.classification(x)
        return x


class MapNet_more_classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MapNet_more_classifier, self).__init__()
        self.in_features = feature_dim

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        self.classification = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.in_features//2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features//2, out_features=num_classes)
        )
        

    def forward(self, x):
        x = F.relu(self.mapping(x))
        x = self.classification(x)
        return x


class MapNet_smaller(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MapNet_smaller, self).__init__()
        self.in_features = feature_dim

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        self.classification = nn.Linear(in_features=self.in_features, out_features=num_classes)
        
    def forward(self, x):
        x = F.relu(self.mapping(x))
        x = self.classification(x)
        return x


class MapNet_bigger_dropout(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MapNet_bigger_dropout, self).__init__()
        self.in_features = feature_dim

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        self.classification = nn.Linear(in_features=self.in_features, out_features=num_classes)
        
    def forward(self, x):
        x = F.relu(self.mapping(x))
        x = self.classification(x)
        return x

class MapNet_supersmall(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MapNet_supersmall, self).__init__()
        self.in_features = feature_dim

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = nn.Linear(in_features=self.in_features, out_features=self.in_features)

        self.classification = nn.Linear(in_features=self.in_features, out_features=num_classes)
        
    def forward(self, x):
        x = F.relu(self.mapping(x))
        x = self.classification(x)
        return x


class MapNet_new(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MapNet_new, self).__init__()
        self.in_features = feature_dim

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        self.classification = nn.Linear(in_features=self.in_features, out_features=num_classes)
        
    def forward(self, x):
        x = F.relu(self.mapping(x))
        x = self.classification(x)
        return x
