import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from typing import List, Any, final, Iterable
import copy


class GradCamBaseModel(nn.Module):
    def __init__(self):
        super(GradCamBaseModel, self).__init__()

        self.to_pad= None

        self.before_conv = nn.Sequential()
        self.pool = nn.Sequential()
        self.after_conv = nn.Sequential()
        self.gradients = None
        self._gradient_list= []

        self._heatmap= []
        self._merge_heatmap = None
        self._pooled_grd= []
        self._activations= None
        self._activations_score= None
        pass

    @abstractmethod
    def forward(self, *args):
        pass

    def reset_gradient_list(self):
        self._gradient_list= []
        pass

    def reset_heatmap_value(self):
        self._heatmap = []
        self._merge_heatmap = None
        pass

    def get_activations(self, x):
        # Reset at each forward step the gradient list
        # IF OVERRIDEN, CALL IT INTO GET_ACTIVATION FUNCTION
        self.reset_gradient_list()

        if self.to_pad is not None:
            x= self.to_pad(x)
        return self.before_conv(x)

    def activations_hook(self, grad):
        # self.gradients = grad
        self._gradient_list.append(grad)
        pass

    def get_activations_gradient(self):
        # return self.gradients
        return self._gradient_list

    def register_hook(self, x):
        return x.register_hook(self.activations_hook)

    def activation_gradient(self, text, num_class) -> List[Any]:
        # Be sure we start the process with a cleared gradient list
        # self.reset_gradient_list()
        # Process the forward step (scoring)
        self.forward(text)[0, num_class].backward()

        gradients= copy.deepcopy(self.get_activations_gradient())
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

    def _compute_heatmap(self, grd, act, type_map, dim):
        polled_grd_element = self.get_pooled_gradient(grd, dim=dim)
        activation_score= self.get_activation_score(act.detach(), polled_grd_element)
        heatmap = self.get_mean_activations_score(activation_score)
        return self._heatmap_mapping(heatmap= heatmap, type_map= type_map)

    def _compute_merged_heatmap(self, gradients, activations, type_map, dim):
        pass

    @staticmethod
    def _heatmap_mapping(heatmap, type_map):

        # TODO: Find other way to adjust the heatmap
        if type_map == "normalized":  # Normalized => [-1, 1]
            heatmap_min = np.min(heatmap)
            heatmap = (2.0 * (heatmap - heatmap_min) / np.ptp(heatmap)) - 1

        elif type_map == "max":  # Relu + /max => [0, 1]
            heatmap = np.maximum(heatmap, 0)
            max_heatmap_value= np.max(heatmap)
            heatmap /= max_heatmap_value
        return heatmap

    def get_heatmap(self, text, num_class, type_map= "normalized", dim= None):
        self.reset_heatmap_value()

        type_map_possible_value= ["normalized", "max"]
        if type_map not in type_map_possible_value:
            raise ValueError(f"Wrong value for the parameter type_map.\nPossible values : {type_map_possible_value}")

        if dim is None:
            dim= [0, 2] # Basic model = CNN Character level

        gradients= self.activation_gradient(text= text, num_class= num_class)
        activations = self.get_activations(text)   #.detach()

        assert len(activations) == len(gradients)

        # Iteration through "filter" gradient feature map (and feature map)
        for i in range(len(gradients)):

            # Ensure activation is 1-D (and so on a Tensor)
            if type(activations) == torch.Tensor:
                act= activations.detach()
            else:
                act= activations[i].detach()

            grd= gradients[i]

            assert grd.size() == act.size()

            heatmap_mapped= self._compute_heatmap(grd= grd, act= act, type_map= type_map, dim= dim)
            self._heatmap.append(heatmap_mapped)


        # Agregate gradient feature and activation map
        if len(self._heatmap) > 1:
            # TODO: Agregate another way (here mean)
            assert len(activations) == len(gradients)
            activations_mean= self.get_pooled_gradient(torch.stack(activations), dim=0) # Simple mean
            gradients_mean = self.get_pooled_gradient(torch.stack(gradients), dim=0) # Simple mean
            assert activations_mean.size() == gradients_mean.size()

            heatmap_merged= self._compute_heatmap(grd= gradients_mean, act= activations_mean
                                                  , type_map= type_map, dim= dim)

            # Last element for _heatmap is the merged result
            self._heatmap.append(heatmap_merged)
            assert len(self._heatmap) == (len(gradients) + 1)

        return copy.deepcopy(self._heatmap)
