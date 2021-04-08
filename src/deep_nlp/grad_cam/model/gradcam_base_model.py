import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from typing import List, Any, final, Iterable
import copy


class GradCamBaseModel(nn.Module):
    """
    This is implementation of `GradCAM <https://arxiv.org/pdf/1610.02391.pdf>`_ for text classification.
    GradCamBaseModel is the class inherited by the model you want to use GradCAM for interpretability.
    For instance, before we had `class CNNCharClassifier(nn.Module):` to declare the pytorch model.
    Now you need `class CNNCharClassifier(GradCamBaseModel):`. **It only works with pytorch**.
    # TODO add those example (for CNN char and Embed CNN)
    Here is a example for the integration of GradCamBaseModel comparative to the orginal model (without legacy).

    It has been developped for 3 kind of CNN based model for NLP classification tasks :

    - `Character level CNN <https://arxiv.org/pdf/1509.01626.pdf>`_
    - `Embedding CNN <https://arxiv.org/pdf/1510.03820.pdf>`_
    - `Embedding with BiLSTM and CNN <https://arxiv.org/pdf/1611.06639.pdf>`_

    Even if it's clearly possible to used it for others CNN architectures.
    """
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
        """
        This method must be implemented by the user, like a normal pytorch model.

        **Warning**, the user need to call `self.get_activations` method inside. Because the philosophy of the
        GradCamBasedModel object it's to separate each step of the model into pipeline like object. And
        `self.get_activations` store the convolution step that generates feature map to study.

        Example :
        --------
        >>> def forward(self, x):
        >>>    x = self.get_activations(x)
        >>>
        >>>    if x.requires_grad:
        >>>        h= self.register_hook(x)

        >>>    x = self.pool(x)

        >>>    x = self.after_conv(x)

        >>>    x = x.view(x.size(0), -1)
        >>>    x= self.fc1(x)
        >>>    x= self.fc2(x)
        >>>    x= self.fc3(x)
        >>>    x= self.log_softmax(x)
        >>>    return x
        """
        pass

    def reset_gradient_list(self):
        """

        @return:
        @rtype:
        """
        self._gradient_list= []
        pass

    def reset_heatmap_value(self):
        """

        @return:
        @rtype:
        """
        self._heatmap = []
        self._merge_heatmap = None
        pass

    def get_activations(self, x):
        """

        @param x:
        @type x:
        @return:
        @rtype:
        """
        # Reset at each forward step the gradient list
        # IF OVERRIDEN, CALL IT INTO GET_ACTIVATION FUNCTION
        self.reset_gradient_list()

        if self.to_pad is not None:
            x= self.to_pad(x)
        return self.before_conv(x)

    def activations_hook(self, grad):
        """

        @param grad:
        @type grad:
        @return:
        @rtype:
        """
        # self.gradients = grad
        self._gradient_list.append(grad)
        pass

    def get_activations_gradient(self):
        """

        @return:
        @rtype:
        """
        # return self.gradients
        return self._gradient_list

    def register_hook(self, x):
        """

        @param x:
        @type x:
        @return:
        @rtype:
        """
        return x.register_hook(self.activations_hook)

    def activation_gradient(self, text, num_class) -> List[Any]:
        """

        @param text:
        @type text:
        @param num_class:
        @type num_class:
        @return:
        @rtype:
        """
        # Be sure we start the process with a cleared gradient list
        # self.reset_gradient_list()
        # Process the forward step (scoring)
        self.forward(text)[0, num_class].backward()

        gradients= copy.deepcopy(self.get_activations_gradient())
        return gradients

    @staticmethod
    def get_pooled_gradient(gradients, dim):
        """

        @param gradients:
        @type gradients:
        @param dim:
        @type dim:
        @return:
        @rtype:
        """
        return torch.mean(gradients, dim= dim)

    @staticmethod
    def get_activation_score(activations, pooled_gradient):
        """

        @param activations:
        @type activations:
        @param pooled_gradient:
        @type pooled_gradient:
        @return:
        @rtype:
        """
        for i in range(activations.shape[1]):
            activations[:, i, :] *= pooled_gradient[i]
        return activations

    @staticmethod
    def get_mean_activations_score(activations_score):
        """

        @param activations_score:
        @type activations_score:
        @return:
        @rtype:
        """
        return torch.mean(activations_score, dim=1).squeeze().numpy()

    def _compute_heatmap(self, grd, act, type_map, dim):
        """

        @param grd:
        @type grd:
        @param act:
        @type act:
        @param type_map:
        @type type_map:
        @param dim:
        @type dim:
        @return:
        @rtype:
        """
        polled_grd_element = self.get_pooled_gradient(grd, dim=dim)
        activation_score= self.get_activation_score(act.detach(), polled_grd_element)
        heatmap = self.get_mean_activations_score(activation_score)
        return self._heatmap_mapping(heatmap= heatmap, type_map= type_map)

    @staticmethod
    def _heatmap_mapping(heatmap: np.ndarray, type_map):
        """
        Normalized heatmap values.

        @param heatmap: an array containing heatmap values
        @type heatmap: np.ndarray
        @param type_map: select the way to normalized. If "max", then heatmap value are going to be between 0, and 1.
        If "normalized", between -1 and 1.
        @type type_map: str
        @return: an array containing normalized heatmap values
        @rtype: np.ndarray

        Example :
        --------
        >>> heatmap= np.array([0.1, -0.5, 0.6, 0, 0, -0.08])
        >>> self._heatmap_mapping(heatmap= heatmap, type_map= "max")
        array([ 0.16666667, -0.83333333,  1.        ,  0.        ,  0.        ,
        -0.13333333])
        """

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
        """

        @param text: input of the model. For instance, for the Embedding CNN model, you enter a tensor of index. And
        for the Character level CNN, you enter a tensor matrix containing 0 and 1.
        @type text:
        @param num_class:
        @type num_class: int
        @param type_map:
        @type type_map: str
        @param dim:
        @type dim: List[int]
        @return:
        @rtype: List[np.ndarray]
        """
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
