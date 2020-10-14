# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os

from .fairseq_optimizer import FairseqOptimizer
from .fp16_optimizer import FP16Optimizer

OPTIMIZER_REGISTRY = {}
OPTIMIZER_CLASS_NAMES = set()


def build_optimizer(args, params):
    if args.task == 'multimodal_translation':
        _params = {}
        _optimizer = {}
        for model_name in params.models.keys():
            _params[model_name] = list(filter(lambda p: p.requires_grad, params.models[model_name].parameters()))
            if model_name == 'en-de' or model_name == 'en-fr':
                _optimizer[model_name] = OPTIMIZER_REGISTRY[args.task1_optimizer](args, _params[model_name], currenttask=model_name)
            elif model_name == 'h5-de' or model_name == 'npz-en':
                _optimizer[model_name] = OPTIMIZER_REGISTRY[args.task2_optimizer](args, _params[model_name], currenttask=model_name)
            else:
                _optimizer[model_name] = OPTIMIZER_REGISTRY[args.disc_optimizer](args, _params[model_name], currenttask=model_name)
        return _optimizer
    else:
        params = list(filter(lambda p: p.requires_grad, params))
        return OPTIMIZER_REGISTRY[args.optimizer](args, params)


def register_optimizer(name):
    """Decorator to register a new optimizer."""

    def register_optimizer_cls(cls):
        if name in OPTIMIZER_REGISTRY:
            raise ValueError('Cannot register duplicate optimizer ({})'.format(name))
        if not issubclass(cls, FairseqOptimizer):
            raise ValueError('Optimizer ({}: {}) must extend FairseqOptimizer'.format(name, cls.__name__))
        if cls.__name__ in OPTIMIZER_CLASS_NAMES:
            # We use the optimizer class name as a unique identifier in
            # checkpoints, so all optimizer must have unique class names.
            raise ValueError('Cannot register optimizer with duplicate class name ({})'.format(cls.__name__))
        OPTIMIZER_REGISTRY[name] = cls
        OPTIMIZER_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_optimizer_cls


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.optim.' + module)
