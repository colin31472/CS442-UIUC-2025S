#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

logger = logging.getLogger(__name__)


class GradReverse(Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        ### YOUR CODES HERE ###
        return x

    @staticmethod
    def backward(ctx, grad_output):
        ### YOUR CODES HERE ###
        return -grad_output


def grad_reverse(x):
    return GradReverse.apply(x)

class FairNet(nn.Module):
    """
    Multi-layer perceptron with adversarial training for fairness.
    """

    def __init__(self, configs):
        super(FairNet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_classes = configs["num_classes"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        # Parameter of the adversary classification layer.
        self.num_adversaries = [self.num_neurons[-1]] + configs["adversary_layers"]
        self.num_adversaries_layers = len(configs["adversary_layers"])
        self.adversaries = nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                          for i in range(self.num_adversaries_layers)])
        self.sensitive_cls = nn.Linear(self.num_adversaries[-1], 2)

    def forward(self, inputs):
        """
        The feature extractor is specified by self.hiddens.
        The label predictor is specified by self.softmax.
        The adversarial discriminator is specified by self.adversaries, followed by self.sensitive_cls.

        You need to return two things:
        1) The first thing is the log of the predicted probabilities (rather than predicted logits) from the label predictor.
        2) The second thing is the log of the predicted probabilities (rather than predicted logits) from the adversarial discriminator.

        Notice:
        For both the label predictor and the adversarial discriminator, we apply the ReLU activation on all layers except for the last linear layer.

        """
        ### YOUR CODES HERE ###
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))                      
        logprobs_y = F.log_softmax(self.softmax(h_relu), dim=1)
        
        adv_relu = grad_reverse(h_relu)
        for adversary in self.adversaries:
            adv_relu = F.relu(adversary(adv_relu))
        logprobs_a = F.log_softmax(self.sensitive_cls(adv_relu), dim=1)
        
        return logprobs_y, logprobs_a

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probability.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        return logprobs


class CFairNet(nn.Module):
    """
    Multi-layer perceptron with adversarial training for conditional fairness.
    """
    def __init__(self, configs):
        super(CFairNet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_classes = configs["num_classes"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        # Parameter of the conditional adversary classification layer.
        self.num_adversaries = [self.num_neurons[-1]] + configs["adversary_layers"]
        self.num_adversaries_layers = len(configs["adversary_layers"])
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for
        # one class label.
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                                         for i in range(self.num_adversaries_layers)])
                                          for _ in range(self.num_classes)])
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 2) for _ in range(self.num_classes)])

    def forward(self, inputs, labels):
        """
        The feature extractor is specified by self.hiddens.
        The label predictor is specified by self.softmax.
        The adversarial discriminator is specified by self.adversaries, followed by self.sensitive_cls.

        You need to return two things:
        1) The first thing is the log of the predicted probabilities (rather than predicted logits) from the label predictor.
        2) The second thing is a list of the log of the predicted probabilities (rather than predicted logits) from the adversarial discriminator,
        where each list corresponds to one class (e.g., Y=0, Y=1, etc)

        Notice:
        For both the label predictor and the adversarial discriminator, we apply the ReLU activation on all layers except for the last linear layer.

        """
        ### YOUR CODES HERE ###
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        logprobs_y = F.log_softmax(self.softmax(h_relu), dim=1)
        
        rev = grad_reverse(h_relu)
        logprobs_a = []
        for i in range(self.num_classes):
            idx_i = (labels == i)
            subset_rev = rev[idx_i]
            adv_relu = subset_rev
            for adversary in self.adversaries[i]:
                adv_relu = F.relu(adversary(adv_relu))
            logprobs_a.append(F.log_softmax(self.sensitive_cls[i](adv_relu), dim=1))
        
        return logprobs_y, logprobs_a

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probabilities.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        return logprobs
