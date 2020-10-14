# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os
from fairseq.modules.ModuleDict import ModuleDict
from .fairseq_lr_scheduler import FairseqLRScheduler


LR_SCHEDULER_REGISTRY = {}


def build_lr_scheduler(args, optimizer):
    if args.task == 'multimodal_translation':
        _lr_scheduler = {}
        for model_name in optimizer.keys():
            if model_name == 'en-de' or model_name == 'en-fr':
                _lr_scheduler[model_name] = LR_SCHEDULER_REGISTRY[args.task1_lr_scheduler](args, optimizer[model_name], model_name)
            elif model_name == 'h5-en' or model_name == 'npz-en':
                _lr_scheduler[model_name] = LR_SCHEDULER_REGISTRY[args.task2_lr_scheduler](args, optimizer[model_name], model_name)
            else:
                _lr_scheduler[model_name] = LR_SCHEDULER_REGISTRY[args.disc_lr_scheduler](args, optimizer[model_name], model_name)
        return _lr_scheduler
    else:
        return LR_SCHEDULER_REGISTRY[args.lr_scheduler](args, optimizer)


def register_lr_scheduler(name):
    """Decorator to register a new LR scheduler."""

    def register_lr_scheduler_cls(cls):
        if name in LR_SCHEDULER_REGISTRY:
            raise ValueError('Cannot register duplicate LR scheduler ({})'.format(name))
        if not issubclass(cls, FairseqLRScheduler):
            raise ValueError('LR Scheduler ({}: {}) must extend FairseqLRScheduler'.format(name, cls.__name__))
        LR_SCHEDULER_REGISTRY[name] = cls
        return cls

    return register_lr_scheduler_cls


# automatically import any Python files in the optim/lr_scheduler/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.optim.lr_scheduler.' + module)
