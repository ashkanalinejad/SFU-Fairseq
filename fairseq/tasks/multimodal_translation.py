# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import os

import torch
from torch.nn import functional
from fairseq import options
from fairseq.data import (
    Dictionary, LanguagePairDataset, IndexedCachedDataset,
    IndexedRawTextDataset, RoundRobinZipDatasets, IndexedDataset,
    AudioDictionary
)
from fairseq.models import FairseqMultiModel

from . import FairseqTask, register_task


@register_task('multimodal_translation')
class MultimodalTranslationTask(FairseqTask):
    """A task for training multiple translation models simultaneously.

    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.

    The training loop is roughly:

        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()

    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.

    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, instead of `--lang-pairs`.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,h5-en')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--audio-input', action='store_true',
                            help='load audio input dataset')
        parser.add_argument('--no-cache-source', default=False, action='store_true')

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        self.langs = list(dicts.keys())
        self.training = training
        self._num_updates = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if not hasattr(args, 'audio_input'):
            args.audio_input = False

        args.lang_pairs = args.lang_pairs.split(',')
        if args.source_lang is not None or args.target_lang is not None:
            #if args.lang_pairs is not None:
            #    raise ValueError(
            #        '--source-lang/--target-lang implies generation, which is '
            #        'incompatible with --lang-pairs'
            #    )
            training = False
            #args.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        else:
            training = True
            #args.lang_pairs = args.lang_pairs.split(',')
            args.source_lang, args.target_lang = args.lang_pairs[0].split('-')

        langs = list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')})

        # load dictionaries
        dicts = OrderedDict()
        for lang in langs:
            if lang == 'h5' or lang == 'npz':
                dicts[lang] = AudioDictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(lang)))
            else:
                dicts[lang] = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[langs[0]].pad()
                assert dicts[lang].eos() == dicts[langs[0]].eos()
                assert dicts[lang].unk() == dicts[langs[0]].unk()
            print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))

        return cls(args, dicts, training)

    def load_dataset(self, split, **kwargs):
        """Load a dataset split."""

        def split_exists(split, src, tgt, lang):
            filename = os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedCachedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedCachedDataset.exists(path):
                return IndexedDataset(path, fix_lua_indexing=True)
            return None

        def sort_lang_pair(lang_pair):
            return '-'.join(sorted(lang_pair.split('-')))

        lang_pairs = self.args.lang_pairs if self.training else ['{}-{}'.format(self.args.source_lang, self.args.target_lang)]
        src_datasets, tgt_datasets = {}, {}
        lang_pairs = lang_pairs if self.args.audio_input else set(map(sort_lang_pair, lang_pairs))
        for lang_pair in lang_pairs:
            src, tgt = lang_pair.split('-')
            if split_exists(split, src, tgt, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif split_exists(split, tgt, src, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                continue
            src_datasets[lang_pair] = indexed_dataset(prefix + src, self.dicts[src])
            tgt_datasets[lang_pair] = indexed_dataset(prefix + tgt, self.dicts[tgt])

            print('| {} {} {} examples'.format(self.args.data, split, len(src_datasets[lang_pair])))

        if len(src_datasets) == 0:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            if lang_pair in src_datasets:
                src_dataset, tgt_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            else:
                lang_pair = sort_lang_pair(lang_pair)
                tgt_dataset, src_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            if self.args.audio_input or src == 'h5' or src == 'npz':
                self.audio_features = src_dataset.sizes[1]
                self.dicts[src].audio_features = self.audio_features
            return LanguagePairDataset(
                src_dataset, src_dataset.sizes, self.dicts[src],
                tgt_dataset, tgt_dataset.sizes, self.dicts[tgt],
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                src_audio=True if src=='h5' or src=='npz' else False
            )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in lang_pairs
            ]),
            eval_key=None if self.training else lang_pairs[0],
        )
        self.lang_pair = lang_pairs[0]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, FairseqMultiModel):
            raise ValueError('MultilingualTranslationTask requires a FairseqMultiModel architecture')

        for model_name in model.models.keys():
            self._num_updates[model_name] = self._num_updates.get(model_name, 0)

        return model

    def enc_dec_train_step(self, sample, model, criterion, optimizer, negative_disc_loss, ignore_grad=False):
        
        lambda_disc = 2

        # set training mode
        for lang_pair in self.args.lang_pairs:
            model.models[lang_pair].train()
        model.models['discriminator'].eval()

        # computing enc-dec loss
        _loss, _sample_size, _logging_output = {}, {}, {}
        for index, lang_pair in enumerate(self.args.lang_pairs):

            if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                continue

            if lang_pair.startswith('h5') or lang_pair.startswith('npz'):
                sample[lang_pair]['net_input']['src_tokens'] = sample[lang_pair]['net_input']['src_tokens'].float()
            sample[lang_pair]['net_input']['lang_pair'] = lang_pair
            gloss, sample_size, logging_output = criterion[lang_pair](model.models[lang_pair], sample[lang_pair])
            
            loss = gloss + (lambda_disc * negative_disc_loss)
            if ignore_grad:
                loss *= 0
            optimizer[lang_pair].backward(loss)

            _loss[lang_pair] = loss.detach().item()
            # TODO make summing of the sample sizes configurable
            _sample_size[lang_pair] = sample_size
            _logging_output[lang_pair] = logging_output
        return _loss, _sample_size, _logging_output

    def discriminator_train_step(self, sample, model, optimizer, ignore_grad=False):
        # set training mode
        for lang_pair in self.args.lang_pairs:
            model.models[lang_pair].eval()
        model.models['discriminator'].train()

        # compute inputs
        encoded = []
        for lang_pair in self.args.lang_pairs:
            src_tokens = sample[lang_pair]['net_input']['src_tokens'].float() if (lang_pair.startswith('h5') or lang_pair.startswith('npz')) else sample[lang_pair]['net_input']['src_tokens']
            encoded.append(model.models[lang_pair].encoder(src_tokens,
                                                           sample[lang_pair]['net_input']['src_lengths'],
                                                           lang_pair))

        dis_inputs = [x['encoder_out'].view(-1, x['encoder_out'].size(-1)) for x in encoded]
        ntokens = [dis_input.size(0) for dis_input in dis_inputs]
        encoded = torch.cat(dis_inputs, 0)
        predictions = model.models['discriminator'](encoded)

        # loss
        self.dis_target = torch.cat([torch.zeros(sz).fill_(i) for i, sz in enumerate(ntokens)])
        self.dis_target = self.dis_target.contiguous().long().cuda()
        y = self.dis_target

        self.dis_target_fake = torch.cat([torch.zeros(sz).fill_(1-i) for i, sz in enumerate(ntokens)])
        self.dis_target_fake = self.dis_target_fake.contiguous().long().cuda()
        neg_y = self.dis_target_fake

        loss = functional.cross_entropy(predictions, y)
        negative_loss = functional.cross_entropy(predictions, neg_y)
        
        if ignore_grad:
            loss *= 0
        optimizer['discriminator'].backward(loss)

        return loss, negative_loss

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            _loss, _sample_size, _logging_output = {}, {}, {}
            for lang_pair in self.args.lang_pairs:
                if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                if lang_pair.startswith('h5') or lang_pair.startswith('npz'):
                    sample[lang_pair]['net_input']['src_tokens'] = sample[lang_pair]['net_input']['src_tokens'].float()
                sample[lang_pair]['net_input']['lang_pair'] = lang_pair
                loss, sample_size, logging_output = criterion[lang_pair](model.models[lang_pair], sample[lang_pair])
                _loss[lang_pair] = loss.data.item()
                # TODO make summing of the sample sizes configurable
                _sample_size[lang_pair] = sample_size
                _logging_output[lang_pair] = logging_output
        return _loss, _sample_size, _logging_output

    def init_logging_output(self, sample):
        return {
            'ntokens': sum(
                sample_lang.get('ntokens', 0)
                for sample_lang in sample.values()
            ) if sample is not None else 0,
            'nsentences': sum(
                sample_lang['target'].size(0) if 'target' in sample_lang else 0
                for sample_lang in sample.values()
            ) if sample is not None else 0,
        }

    def grad_denom(self, sample_sizes, criterion):
        return {lang_pair: criterion[lang_pair].__class__.grad_denom([dict[lang_pair] for dict in sample_sizes])
                for lang_pair in self.args.lang_pairs
                }

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        # aggregate logging outputs for each language pair
        agg_logging_outputs = {
            lang_pair: criterion[lang_pair].__class__.aggregate_logging_outputs([
                logging_output.get(lang_pair, {}) for logging_output in logging_outputs
            ])
            for lang_pair in self.args.lang_pairs
        }

        def sum_over_languages(key):
            return sum(logging_output[key] for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')
        return flat_logging_output

    @property
    def source_dictionary(self):
        return self.dicts[self.args.source_lang]

    @property
    def target_dictionary(self):
        return self.dicts[self.args.target_lang]

    def max_positions(self):
        return (self.args.max_source_positions, self.args.max_target_positions) if not self.training else None
