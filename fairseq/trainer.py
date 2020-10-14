# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Train a network across multiple GPUs.
"""

import os
from collections import OrderedDict
from itertools import chain

import torch
import random

from fairseq import distributed_utils, models, optim, utils
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler
from statistics import mean


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion, dummy_batch):

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        self.args = args
        self.task = task

        # copy model and criterion to current device
        self.criterion = criterion
        if args.fp16:
            self._model = model.half()
        else:
            self._model = model

        self._dummy_batch = dummy_batch
        self._num_updates = {}
        self._optim_history = None
        self._optimizer = None
        self._wrapped_model = None

        for model_name in self._model.models.keys():
            self._num_updates[model_name] = self._num_updates.get(model_name, 0)

        self.init_meters(args)

    def init_meters(self, args):
        self.meter_dict = {}
        for model_name in self._model.models.keys():
            meters = OrderedDict()
            meters['train_loss'] = AverageMeter()
            meters['train_nll_loss'] = AverageMeter()
            meters['discriminator_loss'] = AverageMeter()
            meters['negative_disc_loss'] = AverageMeter()
            meters['valid_loss'] = AverageMeter()
            meters['valid_nll_loss'] = AverageMeter()
            meters['wps'] = TimeMeter()       # words per second
            meters['ups'] = TimeMeter()       # updates per second
            meters['wpb'] = AverageMeter()    # words per batch
            meters['bsz'] = AverageMeter()    # sentences per batch
            meters['gnorm'] = AverageMeter()  # gradient norm
            meters['clip'] = AverageMeter()   # % of updates clipped
            meters['oom'] = AverageMeter()    # out of memory
            if args.fp16:
                meters['loss_scale'] = AverageMeter()  # dynamic loss scale
            meters['wall'] = TimeMeter()      # wall time in seconds
            meters['train_wall'] = StopwatchMeter()  # train wall time in seconds

            self.meter_dict[model_name] = meters


    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size >= 1:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    def _build_optimizer(self):
        if self.args.fp16:
            if torch.cuda.get_device_capability(0)[0] < 7:
                print('| WARNING: your device does NOT support faster training with --fp16, '
                      'please switch to FP32 which is likely to be faster')
            params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if torch.cuda.get_device_capability(0)[0] >= 7:
                print('| NOTICE: your device may support faster training with --fp16')
            self._optimizer = optim.build_optimizer(self.args, self.model)

        self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self._optimizer)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        address, name = filename.rsplit('/', 1)
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            if 'best' in name:
                lang_pair = name.split('_')[1]
                new_filename = os.path.join(address, '{}_{}'.format(lang_pair, name))
                utils.save_state(
                    new_filename, self.args, self.get_model(model_name=lang_pair), self.criterion[lang_pair],
                    self.optimizer[lang_pair], self.lr_scheduler[lang_pair], self._num_updates[lang_pair],
                    self._optim_history, extra_state,
                )
            else:
                for model_key in self.model.models.keys():
                    extra_state['train_meters'] = self.meter_dict[model_key]
                    new_filename = os.path.join(address, '{}_{}'.format(model_key, name))
                    utils.save_state(
                        new_filename, self.args, self.get_model(model_name=model_key), self.criterion[model_key],
                        self.optimizer[model_key], self.lr_scheduler[model_key], self._num_updates[model_key],
                        self._optim_history, extra_state,
                    )


    def load_checkpoint(self, filename, reset_optimizer=False, reset_lr_scheduler=False, optimizer_overrides=None):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = \
            utils.load_model_state(filename, self.get_model())
        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim['criterion_name'] == self.criterion.__class__.__name__, \
                'criterion does not match; please reset the optimizer (--reset-optimizer)'
            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'optimizer does not match; please reset the optimizer (--reset-optimizer)'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters.update(extra_state['train_meters'])
            del extra_state['train_meters']

            # reset TimeMeters, since their start times don't make sense anymore
            for meter in self.meters.values():
                if isinstance(meter, TimeMeter):
                    meter.reset()

        return extra_state

    def train_step(self, samples, dummy_batch=False):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.model.train()
        self.zero_grad()

        if not dummy_batch:
            for lang_pair in self.args.lang_pairs:
                self.meter_dict[lang_pair]['train_wall'].start()

        # Discriminator step
        neg_logging_loss, logging_loss, sample_sizes, disc_loss, neg_disc_loss = [], [], [], 0, 0
        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            try:
                if self.args.distributed_world_size > 1:
                    # Whenever *samples* contains more than one mini-batch, we
                    # want to accumulate gradients locally and only call
                    # all-reduce in the last backwards pass. Currently the
                    # *need_reduction* flag is only supported by
                    # LegacyDistributedDataParallel.
                    if i < len(samples) - 1:
                        self.model.accumulate_grads = True
                    else:
                        self.model.accumulate_grads = False

                # forward and backward
                loss, neg_loss = self.task.discriminator_train_step(
                    sample, self.model, self.optimizer,
                    ignore_grad
                )

                logging_loss.append(loss.detach().item())
                neg_logging_loss.append(neg_loss.detach().item())
            except RuntimeError as e:
#                if 'out of memory' in str(e):
#                    print('| WARNING: ran out of memory, skipping batch')
#                    ooms += 1
#                    self.zero_grad()
#                else:
                raise e

        disc_loss = mean(logging_loss)
        neg_disc_loss = mean( neg_logging_loss )
        if not dummy_batch and random.randint(0,100) < 50:
            try:
                grad_norm = self.optimizer['discriminator'].clip_grad_norm(20)

                # take an optimization step
                self.optimizer['discriminator'].step()
                self._num_updates['discriminator'] += 1

                # update learning rate
                self.lr_scheduler['discriminator'].step_update(self._num_updates['discriminator'])
            except OverflowError as e:
                print('| WARNING: discriminator overflow detected, ' + str(e))
                self.zero_grad(model='discriminator')

        # Encoder-Decoder step
        logging_outputs, sample_sizes, ooms = [], [], 0
        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            try:
                if self.args.distributed_world_size > 1:
                    # Whenever *samples* contains more than one mini-batch, we
                    # want to accumulate gradients locally and only call
                    # all-reduce in the last backwards pass. Currently the
                    # *need_reduction* flag is only supported by
                    # LegacyDistributedDataParallel.
                    if i < len(samples) - 1:
                        self.model.accumulate_grads = True
                    else:
                        self.model.accumulate_grads = False

                # forward and backward
                loss, sample_size, logging_output = self.task.enc_dec_train_step(
                    sample, self.model, self.criterion, self.optimizer, neg_logging_loss[i],
                    ignore_grad
                )
                if not ignore_grad:
                    logging_outputs.append(logging_output)
                    sample_sizes.append(sample_size)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    ooms += 1
                    self.zero_grad()
                else:
                    raise e

        if dummy_batch:
            return None
        logging_output = self.update_param(samples, logging_outputs, sample_sizes, ooms, disc_loss, neg_disc_loss)
        return logging_output

    def update_param(self, samples, logging_outputs, sample_sizes, ooms, disc_loss, neg_disc_loss):
        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_outputs, sample_sizes, ooms = zip(*distributed_utils.all_gather_list(
                [logging_outputs, sample_sizes, ooms],
            ))
            logging_outputs = list(chain.from_iterable(logging_outputs))
            sample_sizes = list(chain.from_iterable(sample_sizes))
            ooms = sum(ooms)

        if ooms == self.args.distributed_world_size * len(samples):
            print('| WARNING: OOM in all workers, skipping `e')
            self.zero_grad()
            return None

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_outputs, self.criterion
        )
        sample_sizes = self.task.grad_denom(sample_sizes, self.criterion)

        if not all(k in logging_output for k in ['ntokens', 'nsentences']):
            raise Exception((
                                'Please update the {}.aggregate_logging_outputs() method to '
                                'return ntokens and nsentences'
                            ).format(self.task.__class__.__name__))

        for index, lang_pair in enumerate(self.args.lang_pairs):

            clip_norm = self.args.task1_clip_norm if index == 0 else self.args.task2_clip_norm

            # normalize grads by sample size
            self.optimizer[lang_pair].multiply_grads(self.args.distributed_world_size / float(sample_sizes[lang_pair]))

            # clip grads
            grad_norm = self.optimizer[lang_pair].clip_grad_norm(clip_norm)

            # take an optimization step
            self.optimizer[lang_pair].step()
            self._num_updates[lang_pair] += 1

            # update learning rate
            self.lr_scheduler[lang_pair].step_update(self._num_updates[lang_pair])

            if not all(k in logging_output for k in ['ntokens', 'nsentences']):
                raise Exception(('Please update the {}.aggregate_logging_outputs() method to '
                                 'return ntokens and nsentences'
                                ).format(self.task.__class__.__name__))

            try:
                # update meters
                ntokens = logging_output.get('{}:ntokens'.format(lang_pair), 0)
                nsentences = logging_output.get('{}:nsentences'.format(lang_pair), 0)
                self.meter_dict[lang_pair]['wps'].update(ntokens)
                self.meter_dict[lang_pair]['ups'].update(1.)
                self.meter_dict[lang_pair]['wpb'].update(ntokens)
                self.meter_dict[lang_pair]['bsz'].update(nsentences)
                self.meter_dict[lang_pair]['gnorm'].update(grad_norm)
                self.meter_dict[lang_pair]['clip'].update(
                    1. if grad_norm > clip_norm and clip_norm > 0 else 0.
                )
                self.meter_dict[lang_pair]['oom'].update(ooms)
                self.meter_dict[lang_pair]['train_loss'].update(logging_output.get('{}:loss'.format(lang_pair), 0), sample_sizes[lang_pair])
                if 'nll_loss' in logging_output:
                    self.meter_dict[lang_pair]['train_nll_loss'].update(logging_output.get('{}:nll_loss'.format(lang_pair), 0), ntokens)
                self.meter_dict[lang_pair]['discriminator_loss'] = disc_loss
                self.meter_dict[lang_pair]['negative_disc_loss'] = neg_disc_loss
            except OverflowError as e:
                print('| WARNING: overflow detected, ' + str(e))
                self.zero_grad()
                logging_output = None

            if self.args.fp16:
                self.meter_dict[lang_pair]['loss_scale'].reset()
                self.meter_dict[lang_pair]['loss_scale'].update(self.optimizer[lang_pair].scaler.loss_scale)

            self.meter_dict[lang_pair]['train_wall'].stop()
        logging_output['disc:loss'] = disc_loss
        logging_output['neg_disc:loss'] = neg_disc_loss
        return logging_output

    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()

            sample = self._prepare_sample(sample)
            if sample is None:
                sample = self._prepare_sample(self._dummy_batch)
                ignore_results = True
            else:
                ignore_results = False

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion
                )
            except RuntimeError as e:
                if 'out of memory' in str(e) and not raise_oom:
                    print('| WARNING: ran out of memory, retrying batch')
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    return self.valid_step(sample, raise_oom=True)
                else:
                    raise e

            if ignore_results:
                logging_output, sample_size = {}, 0

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_output, sample_size = zip(*distributed_utils.all_gather_list(
                [logging_output, sample_size],
            ))
            logging_output = list(logging_output)
 
            # print(sample_size)
            sample_size = list(sample_size)
        else:
            logging_output = [logging_output]
            sample_size = [sample_size]

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_output, self.criterion
        )
        true_sample = []
        for element in sample_size:
            if type(element) == dict:
                true_sample.append(element)
        sample_size = self.task.grad_denom(
            true_sample, self.criterion
        )

        for lang_pair in sample_size.keys():
            # update meters for validation
            ntokens = logging_output.get('{}:ntokens'.format(lang_pair), 0)
            self.meter_dict[lang_pair]['valid_loss'].update(logging_output.get('{}:loss'.format(lang_pair), 0), sample_size[lang_pair])
            if 'nll_loss' in logging_output:
                self.meter_dict[lang_pair]['valid_nll_loss'].update(
                    logging_output.get('{}:nll_loss'.format(lang_pair), 0), ntokens
                )

        return logging_output

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, dummy_batch=True)
        self.zero_grad()

    def zero_grad(self, model=None):
        if model is not None:
            self.optimizer[model].zero_grad()
        else:
            for _, opt in self.optimizer.items():
                opt.zero_grad()

    def lr_step(self, epoch, val_loss=None, model_name=None):
        """Adjust the learning rate based on the validation loss."""
        lr = {}
        if model_name == None:
            for key, Scheduler in self.lr_scheduler.items():
                if key == 'discriminator':
                    try:
                        lr[key] = Scheduler.step(epoch, val_loss['en-de'])
                    except:
                        lr[key] = Scheduler.step(epoch, val_loss['en-fr'])
                else:
                    lr[key] = Scheduler.step(epoch, val_loss[key])
        return lr

    def lr_step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(num_updates)

    def get_lr(self):
        """Get the current learning rate."""
        lr = {}
        for key, opt in self.optimizer.items():
            lr[key] = opt.get_lr()
            lr[key] = eval(lr[key]) if type(lr[key]) == str else lr[key]
        return lr

    def get_model(self, model_name=None):
        """Get the (non-wrapped) model instance."""
        if model_name != None:
            return self._model.models[model_name]
        else:
            return self._model

    def get_meter(self, name):
        """Get a specific meter by name."""
        _meters = {}
        for lang_pair in self._model.models.keys():
            if name not in self.meter_dict[lang_pair]:
                return None
            _meters[lang_pair] = self.meter_dict[lang_pair][name]
        return _meters

    def get_num_updates(self, model_name='discriminator'):
        """Get the number of parameters updates."""
        return self._num_updates[model_name]

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        return utils.move_to_cuda(sample)
