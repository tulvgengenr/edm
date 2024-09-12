# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

@persistence.persistent_class
class EDMLoss_with_ISM:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, ism_weight=0.0, ism_rng_mean=-2.0, ism_dy=1e-5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.ism_weight = ism_weight
        self.ism_rng_mean = ism_rng_mean
        self.ism_dy = ism_dy

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)

        # compute loss_edm
        loss_edm = weight * ((D_yn - y) ** 2)

        # compute loss_ism
        loss_ism = torch.zeros_like(loss_edm)
        if self.ism_weight != 0.0:
            rnd_normal_ism = torch.randn([images.shape[0], 1, 1, 1], device=images.device) + self.ism_rng_mean
            sigma_ism = (rnd_normal_ism * self.P_std + self.P_mean).exp()

            weight_ism = (sigma_ism ** 2 + self.sigma_data ** 2) / (sigma_ism * self.sigma_data) ** 2
            y_ism, augment_labels_ism = augment_pipe(images) if augment_pipe is not None else (images, None)

            n_ism = torch.randn_like(y_ism) * sigma_ism
            D_yn_ism = net(y_ism + n_ism, sigma_ism, labels, augment_labels=augment_labels_ism)

            loss_ism_first = torch.sum(((D_yn_ism - (y_ism)) ** 2))

            y_tilde = y_ism + self.ism_dy
            D_yn_tilde = net(y_tilde + n_ism, sigma_ism, labels, augment_labels=augment_labels_ism)
            nabla_D_yn = (D_yn_tilde - D_yn) / self.ism_dy
            weight_ism_second = 2 * sigma_ism ** 2

            loss_ism_second = weight_ism_second * nabla_D_yn

            loss_ism = self.ism_weight * weight_ism * (loss_ism_first + loss_ism_second)
    
        return loss_edm, loss_ism

#----------------------------------------------------------------------------
