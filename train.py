#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import itertools
import os
import math
import torch

from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def main(args):
    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if not torch.cuda.is_available():
       raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args, model)
    '''
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))
'''
    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    dummy_batch = task.dataset('valid').get_dummy_batch(args.max_tokens, max_positions)

    # Build trainer
    trainer = Trainer(args, task, model, criterion, dummy_batch)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Initialize dataloader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )

    # Load the latest checkpoint if one is available
    if not load_checkpoint(args, trainer, epoch_itr):
        trainer.dummy_train_step([dummy_batch])

    #Freeze encoder weights if requested
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # Train until the learning rate gets too small
    max_epoch = args.task1_max_epoch or math.inf
    max_update = args.task1_max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    while lr[model.keys[0]] > args.task1_min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses)

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""

    # Update parameters every N batches
    if epoch_itr.epoch <= len(args.task1_update_freq):
        update_freq = args.task1_update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.task1_update_freq[-1]


    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(fix_batches_to_gpus=args.fix_batches_to_gpus)
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    max_update = args.task1_max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        stats = collections.OrderedDict()
        for lang_pair in trainer.model.keys:
            # log mid-epoch stats
            stats = get_training_stats(stats, trainer, lang_pair)
            if i == 0:
                trainer.get_meter('wps')[lang_pair].reset()
            for k, v in log_output.items():
                if k in ['{}:loss'.format(lang_pair)]: #, '{}:nll_loss'.format(lang_pair)]:
                    continue  # these are already logged above
                if 'loss' in k:
                    extra_meters[k].update(v, log_output['sample_size'])
                else:
                    extra_meters[k].update(v)
#                stats[k] = extra_meters[k].avg
#        stats['disc_loss'] = log_output['disc:loss']
#        stats['neg_disc_loss'] = log_output['neg_disc:loss']
        progress.log(stats)

            # ignore the first mini-batch in words-per-second calculation

        num_updates = trainer.get_num_updates(lang_pair)
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0:
            valid_losses = validate(args, trainer, task, epoch_itr, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses)

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = collections.OrderedDict()
    for lang_pair in trainer.model.keys:
        stats = get_training_stats(stats, trainer, lang_pair)
#    for k, meter in extra_meters.items():
#        stats[k] = meter.avg
    progress.print(stats)

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        for _, meter in trainer.get_meter(k).items():
            if meter is not None:
                meter.reset()


def get_training_stats(stats, trainer, lang_pair=None):
    stats['{}:loss'.format(lang_pair)] = '{:.3f}'.format(trainer.get_meter('train_loss')[lang_pair].avg)
    if trainer.get_meter('train_nll_loss')[lang_pair].count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')[lang_pair].avg
    #    stats['{}:nll_loss'.format(lang_pair)] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss')[lang_pair].avg
    stats['{}:ppl'.format(lang_pair)] = get_perplexity(nll_loss)
#    stats['wps'] = round(trainer.get_meter('wps')[lang_pair].avg)
#    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups')[lang_pair].avg)
#    stats['wpb'] = round(trainer.get_meter('wpb')[lang_pair].avg)
#    stats['bsz'] = round(trainer.get_meter('bsz')[lang_pair].avg)
    stats['num_updates'] = trainer.get_num_updates(model_name=lang_pair)
    stats['{}:lr'.format(lang_pair)] = trainer.get_lr()[lang_pair]
#    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm')[lang_pair].avg)
#    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip')[lang_pair].avg)
#    stats['oom'] = trainer.get_meter('oom')[lang_pair].avg
#    if trainer.get_meter('loss_scale') is not None:
#        stats['loss_scale'] = '{:.3f}'.format(trainer.get_meter('loss_scale')[lang_pair].avg)
#    stats['wall'] = round(trainer.get_meter('wall')[lang_pair].elapsed_time)
#    stats['train_wall'] = round(trainer.get_meter('train_wall')[lang_pair].sum)
    stats['disc:loss'] = '{:.5f}'.format(trainer.get_meter('discriminator_loss')[lang_pair])
    stats['neg_disc:loss'] = '{:.5f}'.format(trainer.get_meter('negative_disc_loss')[lang_pair])
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = {}
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        for lang_pair in trainer.model.keys:

            # reset validation loss meters
            for k in ['valid_loss', 'valid_nll_loss']:
                meter = trainer.get_meter(k)[lang_pair]
                if meter is not None:
                    meter.reset()
            extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

        stats = collections.OrderedDict()
        for lang_pair in trainer.model.keys:
            for k, v in log_output.items():
                if k in ['{}:loss'.format(lang_pair), '{}:nll_loss'.format(lang_pair)]:
                    continue
                if 'loss' in k:
                    extra_meters[k].update(v, log_output['sample_size'])
                else:
                    extra_meters[k].update(v)

            # log validation stats
            stats = get_valid_stats(stats, trainer, lang_pair)
            valid_losses[lang_pair] = stats['{}:valid_loss'.format(lang_pair)]

        progress.print(stats)

    return valid_losses


def get_valid_stats(stats, trainer, lang_pair=None):
    if lang_pair != 'discriminator':
        stats['{}:valid_loss'.format(lang_pair)] = trainer.get_meter('valid_loss')[lang_pair].avg
        if trainer.get_meter('valid_nll_loss')[lang_pair].count > 0:
            nll_loss = trainer.get_meter('valid_nll_loss')[lang_pair].avg
            stats['{}:valid_nll_loss'.format(lang_pair)] = nll_loss
        else:
            nll_loss = trainer.get_meter('valid_loss')[lang_pair].avg
        stats['{}:valid_ppl'.format(lang_pair)] = get_perplexity(nll_loss)
        stats['num_updates'] = trainer.get_num_updates(model_name=lang_pair)
        if hasattr(save_checkpoint, 'best'):
            stats['{}:best'.format(lang_pair)] = min(save_checkpoint.best[lang_pair], stats['{}:valid_loss'.format(lang_pair)])

    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    for lang_pair in trainer.model.models.keys():
        if lang_pair != 'discriminator':
            checkpoint_conds['checkpoint_{}_best.pt'.format(lang_pair)] = (
                    val_loss is not None and
                    (not hasattr(save_checkpoint, 'best') or val_loss[lang_pair] < save_checkpoint.best[lang_pair])
            )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    save_checkpoint.best = {}
    for k, v in val_loss.items():
        if val_loss is not None:
            save_checkpoint.best[k] = min(v, prev_best[k])
    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path, args.reset_optimizer, args.reset_lr_scheduler,
                                              eval(args.optimizer_overrides))
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']
        return True
    return False


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e


if __name__ == '__main__':
    parser = options.get_training_parser(default_task='multimodal_pretraining')
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)
