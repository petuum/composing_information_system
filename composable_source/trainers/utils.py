#  Copyright 2021 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import torch
import torch.nn.functional as F
import texar.torch as tx
from forte.utils import get_class


def get_lr_multiplier(step: int,
                      total_steps: int,
                      warmup_steps: int) -> float:
    r"""Calculate the learning rate multiplier given current step
    and the number of warm-up steps. The learning rate schedule
    follows a linear warm-up and linear decay.
    """
    step = min(step, total_steps)
    multiplier = (1 - (step - warmup_steps) /
                (total_steps - warmup_steps))
    if warmup_steps > 0 and step < warmup_steps:
        warmup_percent_done = step / warmup_steps
        multiplier = warmup_percent_done
    return multiplier


# Build learning rate decay scheduler.
def build_lr_decay_scheduler(model,
                             num_train_data: int,
                             train_batch_size: int,
                             num_epochs: int,
                             warmup_proportion: float):
    """Built Bert model learning rate decay scheduler."""
    num_train_steps = int(num_train_data /
                        train_batch_size *
                        num_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    static_lr = 2e-5
    vars_with_decay = []
    vars_without_decay = []
    for name, param in model.named_parameters():
        if 'layer_norm' in name or name.endswith('bias'):
            vars_without_decay.append(param)
        else:
            vars_with_decay.append(param)

    opt_params = [{
        'params': vars_with_decay,
        'weight_decay': 0.01,
    }, {
        'params': vars_without_decay,
        'weight_decay': 0.0,
    }]
    optim = tx.core.BertAdam(
        opt_params, betas=(0.9, 0.999), eps=1e-6, lr=static_lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, functools.partial(get_lr_multiplier,
                                total_steps=num_train_steps,
                                warmup_steps=num_warmup_steps))
    return scheduler, optim


def compute_loss(model, logits, labels):
    r"""Compute loss.
    """
    if model.is_binary:
        loss = F.binary_cross_entropy(
            logits.view(-1), labels.view(-1), reduction='mean')
    else:
        loss = F.cross_entropy(
            logits.view(-1, model.num_classes),
            labels.view(-1), reduction='mean')
    return loss


def create_class(class_name, class_config):
    class_instance = get_class(class_name)()
    return class_instance, class_config
