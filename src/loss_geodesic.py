import torch
import numpy as np
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.geometry.riemannian_metric as riem
SE3_GROUP = SpecialEuclidean(n=3, point_type='vector')
METRIC = SE3_GROUP.left_canonical_metric

# pure geodesic loss function, calculate the geodesic distance between two twist
class Geodesic_loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predict, target):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # predict: [n_samples, 6]
        # target: [n_samples, 6]
        # output: [1] average loss between input and target
        ctx.save_for_backward(predict, target)
        predict = predict.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        loss = riem.loss(predict, target, METRIC)
        loss = torch.tensor(loss, dtype=torch.float32, device="cuda")
        avg_loss = torch.mean(loss, dim=0, keepdim=True)

        return avg_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        predict, target = ctx.saved_tensors
        grad = riem.grad(predict.detach().cpu(), target.detach().cpu(), METRIC)
        grad = torch.tensor(grad, dtype=torch.float32, device="cuda")

        return grad, None
