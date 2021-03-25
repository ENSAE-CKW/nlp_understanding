import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod


class GradCamBaseModel(nn.Module):
    def __init__(self):
        super(GradCamBaseModel, self).__init__()

        self.to_pad= None

        self.before_conv = nn.Sequential()
        self.pool = nn.Sequential()
        self.after_conv = nn.Sequential()
        self.gradients = None

        self._heatmap= None
        self._pooled_grd= None
        self._activations= None
        self._activations_score= None
        pass

    @abstractmethod
    def forward(self, *args):
        pass

    def get_activations(self, x):
        if self.to_pad is not None:
            x= self.to_pad(x)
        return self.before_conv(x)

    def activations_hook(self, grad):
        self.gradients = grad
        pass

    def get_activations_gradient(self):
        return self.gradients

    def register_hook(self, x):
        return x.register_hook(self.activations_hook)

    def activation_gradient(self, text, num_class):

        self.forward(text)[0, num_class].backward()

        gradients= self.get_activations_gradient()
        return gradients

    @staticmethod
    def get_pooled_gradient(gradients, dim):
        return torch.mean(gradients, dim= dim)

    @staticmethod
    def get_activation_score(activations, pooled_gradient):
        for i in range(activations.shape[1]):
            activations[:, i, :] *= pooled_gradient[i]
        return activations

    @staticmethod
    def get_mean_activations_score(activations_score):
        return torch.mean(activations_score, dim=1).squeeze().numpy()

    def get_heatmap(self, text, num_class, type= "normalized", dim= None):

        if type not in ["normalized", "max"]:
            raise ValueError()

        if dim is None:
            dim= [0, 2]

        gradients= self.activation_gradient(text= text, num_class= num_class)

        self._pooled_grd = self.get_pooled_gradient(gradients, dim= dim)

        self._activations = self.get_activations(text).detach()

        self._activations_score= self.get_activation_score(self._activations , self._pooled_grd)

        self._heatmap = self.get_mean_activations_score(self._activations_score)

        if type == "normalized":
            heatmap_min = np.min(self._heatmap)
            heatmap = (2.0 * (self._heatmap - heatmap_min) / np.ptp(self._heatmap)) - 1

        elif type == "max":
            heatmap = np.maximum(self._heatmap, 0)
            heatmap /= np.max(heatmap)

        return heatmap
