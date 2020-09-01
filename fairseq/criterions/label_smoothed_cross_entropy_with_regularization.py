# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('label_smoothed_cross_entropy_with_regularization')
class LabelSmoothedCrossEntropyCriterionWithRegularization(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, regularization_weight):
        super().__init__(task, sentence_avg, label_smoothing)
        self.regularization_weight = regularization_weight

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--regularization_weight', default=0.01, type=float, metavar='D',
                            help='weight for the regularization loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        primary_net_output = model(**sample['primary']['net_input'])
        primary_loss, primary_nll_loss = self.compute_loss(model, primary_net_output, sample['primary'], reduce=reduce)
        primary_sample_size = sample['primary']['target'].size(0) if self.sentence_avg else sample['primary']['ntokens']

        secondary_net_output = model(**sample['secondary']['net_input'])
        secondary_loss, secondary_nll_loss = self.compute_loss(model, secondary_net_output, sample['secondary'], reduce=reduce)
        secondary_sample_size = sample['secondary']['target'].size(0) if self.sentence_avg else sample['secondary']['ntokens']

        regularization_loss = self.compute_regularization_loss(primary_net_output, secondary_net_output)

        loss = primary_loss + secondary_loss + self.regularization_weight * regularization_loss
        nll_loss = primary_nll_loss + secondary_nll_loss
        ntokens = sample['primary']['ntokens'] + sample['secondary']['ntokens']
        nsentences = sample['primary']['target'].size(0) + sample['secondary']['target'].size(0)
        sample_size = primary_sample_size + secondary_sample_size

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'regularization_loss': utils.item(regularization_loss.data) if reduce else regularization_loss.data,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        return loss, sample_size, logging_output

    def compute_regularization_loss(self, model, primary_net_output, secondary_net_output):
        mean_net_output = (primary_net_output + secondary_net_output) / 2
        m = model.get_normalized_probs(mean_net_output, log_probs=False)
        p = model.get_normalized_probs(primary_net_output, log_probs=True)
        q = model.get_normalized_probs(secondary_net_output, log_probs=True)
        loss = (F.kl_div(p, m) + F.kl_div(q, m)) / 2
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        regularization_loss_sum = utils.item(sum(log.get('regularization_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('regularization_loss', regularization_loss_sum / sample_size, sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
