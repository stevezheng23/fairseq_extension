# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
import itertools
import logging
import os

import numpy as np

import torch

from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.tasks import FairseqTask, register_task

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
    num_buckets=0,
    shuffle=True,
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset, eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
    )


@register_task('translation')
class TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')

        # options for task-specific data augmentation
        parser.add_argument('--augmentation_schema', default='cut_off',  type=str,
                            help='augmentation schema: e.g. `cut_off`')
        parser.add_argument('--augmentation_masking_schema', default='word',  type=str,
                            help='augmentation masking schema: e.g. `word`, `span`')
        parser.add_argument('--augmentation_masking_probability', default=0.15, type=float,
                            help='augmentation masking probability')
        parser.add_argument('--augmentation_replacing_schema', default=None,  type=str,
                            help='augmentation replacing schema: e.g. `mask`, `random`, `mixed`')
        parser.add_argument("--augmentation_span_len_dist", default='geometric', type=str,
                            help="augmentation span length distribution e.g. geometric, poisson, etc.")
        parser.add_argument("--augmentation_max_span_len", type=int, default=10,
                            help="augmentation maximum span length")
        parser.add_argument("--augmentation_min_num_spans", type=int, default=5,
                            help="augmentation minimum number of spans")
        parser.add_argument("--augmentation_geometric_prob", type=float, default=0.2,
                            help="augmentation probability of geometric distribution.")
        parser.add_argument("--augmentation_poisson_lambda", type=float, default=5.0,
                            help="augmentation lambda of poisson distribution.")

        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != 'test'),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    def augment_sample(self, sample):
        augmented_sample = {
            'id': sample['id'].clone(),
            'nsentences': sample['nsentences'].clone(),
            'ntokens': sample['ntokens'].clone(),
            'net_input': {
                'src_tokens': None,
                'src_lengths': sample['net_input']['src_lengths'].clone(),
                'prev_output_tokens': None,
            },
            'target': sample['target'].clone()
        }

        if self.args.augmentation_schema == 'cut_off':
            augmented_sample['net_input']['src_tokens'] = self._mask_tokens(sample['net_input']['src_tokens'], self.src_dict)
            augmented_sample['net_input']['prev_output_tokens'] = self._mask_tokens(sample['net_input']['prev_output_tokens'], self.tgt_dict)
        else:
            raise ValueError("Augmentation schema {0} is not supported".format(self.args.augmentation_schema))

        augmented_sample = {
            'id': torch.cat((sample['id'], augmented_sample['id']), dim=0),
            'nsentences': torch.cat((sample['nsentences'], augmented_sample['nsentences']), dim=0),
            'ntokens': torch.cat((sample['ntokens'], augmented_sample['ntokens']), dim=0),
            'net_input': {
                'src_tokens': torch.cat((sample['net_input']['src_tokens'], augmented_sample['net_input']['src_tokens']), dim=0),
                'src_lengths': torch.cat((sample['net_input']['src_lengths'], augmented_sample['net_input']['src_lengths']), dim=0),
                'prev_output_tokens': torch.cat((sample['net_input']['prev_output_tokens'], augmented_sample['net_input']['prev_output_tokens']), dim=0),
            },
            'target': torch.cat((sample['target'], augmented_sample['target']), dim=0)
        }

        return augmented_sample

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def _mask_tokens(self, inputs):
        if self.args.augmentation_masking_schema == 'word':
            masked_inputs = self._mask_tokens_by_word(inputs)
        elif self.args.augmentation_masking_schema == 'span':
            masked_inputs = self._mask_tokens_by_span(inputs)
        else:
            raise ValueError("The masking schema {0} is not supported".format(self.args.augmentation_masking_schema))

        return masked_inputs

    def _mask_tokens_by_word(self, inputs, vocab_dict):
        vocab_size = len(vocab_dict)
        bos_index, eos_index = vocab_dict.bos(), vocab_dict.eos()
        pad_index, unk_index = vocab_dict.pad(), vocab_dict.unk()

        available_token_indices = (inputs != bos_index) & (inputs != eos_index) & (inputs != pad_index) & (inputs != unk_index)
        random_masking_indices = torch.bernoulli(torch.full(inputs.shape, self.args.augmentation_masking_probability, device=inputs.device)).bool()

        masked_inputs = inputs.clone()
        masking_indices = random_masking_indices & available_token_indices
        self._replace_token(masked_inputs, masking_indices, unk_index, vocab_size)

        return masked_inputs

    def _mask_tokens_by_span(self, inputs, vocab_dict):
        vocab_size = len(vocab_dict)
        bos_index, eos_index = vocab_dict.bos(), vocab_dict.eos()
        pad_index, unk_index = vocab_dict.pad(), vocab_dict.unk()

        span_info_list = self.generate_spans(inputs)

        num_spans = len(span_info_list)
        masked_span_list = np.random.binomial(1, self.args.augmentation_masking_probability, size=num_spans).astype(bool)
        masked_span_list = [span_info_list[i] for i, masked in enumerate(masked_span_list) if masked]

        available_token_indices = (inputs != bos_index) & (inputs != eos_index) & (inputs != pad_index) & (inputs != unk_index)
        random_masking_indices = torch.zeros_like(inputs)
        for batch_index, seq_index, span_length in masked_span_list:
            random_masking_indices[batch_index, seq_index:seq_index + span_length] = 1

        masked_inputs = inputs.clone()
        masking_indices = random_masking_indices.bool() & available_token_indices
        self._replace_token(masked_inputs, masking_indices, unk_index, vocab_size)

        return masked_inputs

    def _sample_span_length(self, span_len_dist, max_span_len, geometric_prob=0.2, poisson_lambda=5.0):
        if span_len_dist == 'geometric':
            span_length = min(np.random.geometric(geometric_prob) + 1, max_span_len)
        elif span_len_dist == 'poisson':
            span_length = min(np.random.poisson(poisson_lambda) + 1, max_span_len)
        else:
            span_length = np.random.randint(max_span_len) + 1

        return span_length

    def _get_default_spans(self, batch_index, seq_length, num_spans):
        span_length = int((seq_length - 2) / num_spans)
        last_span_length = seq_length - 2 - (num_spans - 1) * span_length
        span_infos = []
        for i in range(num_spans):
            span_info = (batch_index, 1 + i * span_length, span_length if i < num_spans - 1 else last_span_length)
            span_infos.append(span_info)

        return span_infos

    def _generate_spans(self, inputs):
        batch_size, seq_length = inputs.size()[0], inputs.size()[1]

        span_info_list = []
        for batch_index in range(batch_size):
            span_infos = []
            seq_index = 1
            max_index = seq_length - 2
            while seq_index <= max_index:
                span_length = self._sample_span_length(self.args.augmentation_span_len_dist,
                    self.args.augmentation_max_span_len, self.args.augmentation_geometric_prob, self.args.augmentation_poisson_lambda)
                span_length = min(span_length, max_index - seq_index + 1)

                span_infos.append((batch_index, seq_index, span_length))
                seq_index += span_length

            if len(span_infos) < self.args.augmentation_min_num_spans:
                span_infos = self._get_default_spans(batch_index, seq_length, self.args.augmentation_min_num_spans)

            span_info_list.extend(span_infos)

        return span_info_list

    def _replace_token(self, inputs, masking_indices, mask_index, vocab_size):
        if self.args.augmentation_replacing_schema == 'mask':
            inputs[masking_indices] = mask_index
        elif self.args.augmentation_replacing_schema == 'random':
            random_words = torch.randint(vocab_size, inputs.shape, device=inputs.device, dtype=torch.long)
            inputs[masking_indices] = random_words[masking_indices]
        elif self.args.augmentation_replacing_schema == 'mixed':
            # 80% of the time, we replace masked input tokens with <unk> token
            mask_token_indices = torch.bernoulli(torch.full(inputs.shape, 0.8, device=inputs.device)).bool() & masking_indices
            inputs[mask_token_indices] = mask_index

            # 10% of the time, we replace masked input tokens with random word
            random_token_indices = torch.bernoulli(torch.full(inputs.shape, 0.5, device=inputs.device)).bool() & masking_indices & ~mask_token_indices
            random_words = torch.randint(vocab_size, inputs.shape, device=inputs.device, dtype=torch.long)
            inputs[random_token_indices] = random_words[random_token_indices]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        else:
            raise ValueError("The replacing schema: {0} is not supported. Only support ['mask', 'random', 'mixed']".format(self.args.augmentation_replacing_schema))
