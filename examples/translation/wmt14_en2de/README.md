# Neural Machine Translation

## Training a new model

### WMT'14 English to German (Transformers)

The following instructions can be used to train a Transformer-based model on the WMT English to German dataset.

The WMT English to German dataset can be preprocessed using the `prepare-wmt14en2de.sh` script.
By default it will produce a dataset that was modeled after [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), but with additional news-commentary-v12 data from WMT'17.
To use only data available in WMT'14, please use the `--icml17` option.

```bash
# Download and prepare the data
cd examples/translation/
# WMT'17 data:
# bash prepare-wmt14en2de.sh
# WMT'14 data:
bash prepare-wmt14en2de.sh --icml17
cd ../..

# Binarize the dataset
TEXT=examples/translation/wmt14_en_de
python fairseq_cli/preprocess.py \
--source-lang en --target-lang de \
--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
--destdir data-bin/wmt14_en_de --thresholdtgt 0 --thresholdsrc 0 \
--workers 20 --joined_dictionary

# Train the model
mkdir -p results/transformer_base_wmt14_en2de
TEXT=results/transformer_base_wmt14_en2de
# launch distributed trainin
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 fairseq_cli/train.py data-bin/wmt14_en_de \
--arch transformer_wmt_en_de \
--share-all-embeddings \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--adam-eps 1e-9 \
--clip-norm 0.0 \
--weight-decay 0.0 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 8192 \
--dropout 0.1 \
--lr-scheduler inverse_sqrt \
--lr 7e-4 \
--warmup-updates 8000 \
--max-epoch 100 \
--update-freq 1 \
--distributed-world-size 4 \
--ddp-backend=c10d \
--keep-last-epochs 50 \
--log-format tqdm \
--log-interval 100 \
--save-dir $TEXT \
--seed 97926458

# Evaluate
# average last 5 checkpoints
python scripts/average_checkpoints.py \
--inputs $TEXT \
--num-epoch-checkpoints 5 \
--output $TEXT/checkpoint_last5.pt
# generate results & quick evaluate
LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py data-bin/wmt14_en_de \
--path $TEXT/checkpoint_last5.pt \
--beam 5 --remove-bpe --lenpen 0.3 >> $TEXT/checkpoint_last5.gen
# compound split & re-run evaluate
bash compound_split_bleu.sh $TEXT/checkpoint_last5.gen
LC_ALL=C.UTF-8 python fairseq_cli/score.py \
--sys $TEXT/checkpoint_last5.gen.sys \
--ref $TEXT/checkpoint_last5.gen.ref
```
