# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.moco import moco_loss_func
# from solo.losses.simclr import simclr_loss_func
# from solo.methods.base import  BaseDistillationMethod
from solo.methods.base_for_adversarial_training import BaseDistillationATMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.misc import gather

from torchvision import models
from solo.utils.metrics import accuracy_at_k, weighted_mean


class MoCoV2KDAT(BaseDistillationATMethod):
    queue: torch.Tensor

    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        queue_size: int,
        teacher_logit_fix: bool,
        loss_type: str,
        projector_ablation: str,
        epsilon: int = 8,
        num_steps: int = 5,
        step_size: int = 2,
        trades_k: float = 3, 
        aux_data: bool = False,
        augmentation_ablation: bool = False,
        expriment_code: str = "000",
        **kwargs
    ):
        """Implements MoCo V2+ (https://arxiv.org/abs/2011.10566).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            queue_size (int): number of samples to keep in the queue.
        """

        super().__init__(**kwargs)

        self.temperature = temperature
        self.queue_size = queue_size
        self.loss_type = loss_type
        self.projector_ablation = projector_ablation
        self.trades_k = trades_k
        self.aux_data = aux_data

        self.augmentation_ablation = augmentation_ablation
        self.expriment_code = expriment_code


        self.epsilon = epsilon/255.
        self.num_steps = num_steps
        self.step_size = step_size/255.


        self.projector = nn.Identity()
        self.momentum_projector = nn.Identity()


    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(MoCoV2KDAT, MoCoV2KDAT).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("mocov2_kd_at")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        # queue settings
        parser.add_argument("--queue_size", default=65536, type=int)


        parser.add_argument("--teacher_logit_fix", action="store_true")

        # choose loss to train the model
        SUPPORT_LOSS = ["ce2gt_ae2ce",
                        "ae2gt_ae2ce",
                        "ce2gt_ae2ce_infonce",
                        "pgd",
                        "ae2gt_infonce",

                        "ae2gt_kl",
                        "ce2gt_ae2ce_kl",
                        "cs.ce2gt._kl.ae2ce.",
                        "kl.ce2gt._cs.ae2ce.",
                        "cs.ce2gt._infonce.ae2ce.",
                        "infonce.ce2gt._cs.ae2ce.",
                        "infonce.ce2gt._kl.ae2ce.",
                        "kl.ce2gt._infonce.ae2ce."
                        ]

        parser.add_argument("--loss_type", default="ce2gt_ae2ce", type=str, choices=SUPPORT_LOSS)


        # projector exploration
        PROJECTOR_ABLATION = ["remove", "same", "correspond", 'only_student', 'only_teacher']
        parser.add_argument("--projector_ablation", default="remove", type=str, choices=PROJECTOR_ABLATION)


        # training adversarial hyper parameter
        parser.add_argument("--epsilon", type=int, default=8)
        parser.add_argument("--step_size", type=int, default=2)
        parser.add_argument("--num_steps", type=int, default=5)

        # for loss factor
        parser.add_argument("--trades_k", type=float, default=2)

        # for augmentation ablation study
        parser.add_argument("--augmentation_ablation", action="store_true")
        parser.add_argument("--expriment_code", default="00", type=str)
        




        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters together with parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs


    def forward(self, X: torch.Tensor):
        """Performs the forward pass of the online backbone and projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        out = self.backbone(X)

        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for MoCo reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the
                format of [img_indexes, [X], Y], where [X] is a list of size self.num_large_crops
                containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MOCO loss and classification loss.

        """


        self.momentum_backbone.eval()


        # import ipdb; ipdb.set_trace()
        if self.aux_data:
            image_tau1, image_weak = batch[0]
            targets = batch[1]

        else:
            image_tau1, image_weak = batch[1]
            targets = batch[2]


        ############################################################################
        # Adversarial Training (CAT)
        ############################################################################
        # if self.trainer.current_epoch ==2:
        #     import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()

        away_target = self.momentum_projector(self.momentum_backbone(image_weak))

        AE_generation_image = image_weak
        image_AE = self.generate_training_AE(AE_generation_image, away_target)

        image_CAT = torch.cat([image_weak, image_AE])
        logits_all = self.projector(self.backbone(image_CAT))
        bs = image_weak.size(0)
        student_logits_clean = logits_all[:bs]
        student_logits_AE = logits_all[bs:]

        # Cosine Similarity loss
        adv_loss = -F.cosine_similarity(student_logits_clean, away_target).mean()
        adv_loss += -self.trades_k*F.cosine_similarity(student_logits_AE, student_logits_clean).mean()

        ############################################################################
        # Adversarial Training (CAT)
        ############################################################################









        ############################################################################
        # Online clean classifier training
        ############################################################################
        # Bug Fix: train classifier using evaluation mode
        self.backbone.eval()
        # import ipdb; ipdb.set_trace()
        outs_image_weak = self._base_shared_step(image_weak, targets)
        self.backbone.train()
        metrics = {
            "train_class_loss": outs_image_weak["loss"],
            "train_acc1": outs_image_weak["acc1"],
            "train_acc5": outs_image_weak["acc5"],
        }
        class_loss_clean = outs_image_weak["loss"]
        self.log_dict(metrics, on_epoch=True)
        ############################################################################
        # Online clean classifier training
        ############################################################################


        ae_std = F.normalize(student_logits_AE, dim=-1).std(dim=0).mean()
        clean_std = F.normalize(student_logits_clean, dim=-1).std(dim=0).mean()
        teacher_std = F.normalize(away_target, dim=-1).std(dim=0).mean()
        
        
        metrics = {
            "adv_loss": adv_loss,
            "ae_std": ae_std,
            "clean_std": clean_std,
            "teacher_std": teacher_std
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        # return adv_loss + class_loss_adv + class_loss_clean
        return adv_loss + class_loss_clean



    def generate_training_AE(self, image: torch.Tensor, away_target: torch.Tensor):
        """
        images_org: weak aug
        away_target: from teacher
        """

        # self.epsilon = 8/255.
        # self.num_steps = 5
        # self.step_size = 2/255.

        x_cl = image.clone().detach()

        # if self.rand:
        x_cl = x_cl + torch.zeros_like(image).uniform_(-self.epsilon, self.epsilon)

        # f_ori_proj = self.model(images_org).detach()
        # Change the attack process of model to eval
        self.backbone.eval()

        for i in range(self.num_steps):
            x_cl.requires_grad_()
            with torch.enable_grad():
                f_proj = self.projector(self.backbone(x_cl))

                # for 16 bit training
                loss_contrast = -F.cosine_similarity(f_proj, away_target, dim=1).sum() *256
                loss = loss_contrast

            # import ipdb ;ipdb.set_trace()
            grad_x_cl = torch.autograd.grad(loss, x_cl)[0]
            # grad_x_cl = torch.autograd.grad(loss, x_cl, grad_outputs=torch.ones_like(loss))[0]
            x_cl = x_cl.detach() + self.step_size * torch.sign(grad_x_cl.detach())

            # remove the clamp in for the image comparision
            x_cl = torch.min(torch.max(x_cl, image - self.epsilon), image + self.epsilon)
            x_cl = torch.clamp(x_cl, 0, 1)

        self.backbone.train()

        return x_cl






