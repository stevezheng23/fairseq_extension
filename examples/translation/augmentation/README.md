# Neural Machine Translation with Data Augmentation

## NMT results

### IWSLT'14 German to English (Transformers)
| Task          | Setting           | Approach   | BLEU     |
|---------------|-------------------|------------|----------|
| iwslt14 de-en | transformer-small | w/o cutoff | 36.2     |
| iwslt14 de-en | transformer-small | w/ cutoff  | **37.6** |

### WMT'14 English to German (Transformers)

| Task          | Setting           | Approach   | BLEU     |
|---------------|-------------------|------------|----------|
| wmt14 en-de   | transformer-base  | w/o cutoff | 28.6     |
| wmt14 en-de   | transformer-base  | w/ cutoff  | 29.1     |
| wmt14 en-de   | transformer-big   | w/o cutoff | 29.5     |
| wmt14 en-de   | transformer-big   | w/ cutoff  | **30.3** |

## NMT experiments

### IWSLT'14 German to English (Transformers)
The following instructions can be used to train a Transformer-based model on the IWSLT'14 German to English dataset.

The IWSLT'14 German to English dataset can be preprocessed using the `prepare-iwslt14.sh` script.

```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Binarize the dataset
DATA=examples/translation/iwslt14.tokenized.de-en
BIN=data-bin/iwslt14.tokenized.de-en
python fairseq_cli/preprocess.py \
--source-lang de --target-lang en \
--trainpref $DATA/train --validpref $DATA/valid --testpref $DATA/test \
--destdir $BIN --thresholdtgt 0 --thresholdsrc 0 \
--workers 20 --joined_dictionary

# Train the model w/o cutoff
RESULT=results/transformer_small_iwslt14_de2en
mkdir -p $RESULT
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
fairseq_cli/train.py $BIN \
--arch transformer_iwslt_de_en \
--share-all-embeddings \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--adam-eps 1e-9 \
--clip-norm 0.0 \
--weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 4096 \
--dropout 0.3 \
--attention-dropout 0.1 \
--activation-dropout 0.1 \
--lr-scheduler inverse_sqrt \
--lr 7e-4 \
--warmup-updates 6000 \
--max-epoch 100 \
--update-freq 1 \
--distributed-world-size 4 \
--ddp-backend=c10d \
--keep-last-epochs 20 \
--log-format tqdm \
--log-interval 100 \
--save-dir $RESULT \
--seed 2128977

# Train the model w/ cutoff
RESULT=results/transformer_small_iwslt14_de2en_cutoff
mkdir -p $RESULT
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
fairseq_cli/train.py $BIN \
--arch transformer_iwslt_de_en \
--augmentation \
--augmentation_schema cut_off \
--augmentation_masking_schema word \
--augmentation_masking_probability 0.05 \
--augmentation_replacing_schema mask \
--share-all-embeddings \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--adam-eps 1e-9 \
--clip-norm 0.0 \
--weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy_with_regularization \
--regularization_weight 5.0 \
--label-smoothing 0.1 \
--max-tokens 2048 \
--dropout 0.3 \
--attention-dropout 0.1 \
--activation-dropout 0.1 \
--lr-scheduler inverse_sqrt \
--lr 7e-4 \
--warmup-updates 6000 \
--max-epoch 100 \
--update-freq 1 \
--distributed-world-size 4 \
--ddp-backend=c10d \
--keep-last-epochs 20 \
--log-format tqdm \
--log-interval 100 \
--save-dir $RESULT \
--seed 40199672

# Evaluate
RESULT=results/transformer_small_iwslt14_de2en
# RESULT=results/transformer_small_iwslt14_de2en_cutoff
# average last 5 checkpoints
python scripts/average_checkpoints.py \
--inputs $RESULT \
--num-epoch-checkpoints 5 \
--output $RESULT/checkpoint_last5.pt
# generate results & quick evaluate
LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py $BIN \
--path $RESULT/checkpoint_last5.pt \
--beam 5 --remove-bpe --lenpen 0.5 >> $RESULT/checkpoint_last5.gen
# compound split & re-run evaluate
bash compound_split_bleu.sh $RESULT/checkpoint_last5.gen
LC_ALL=C.UTF-8 python fairseq_cli/score.py \
--sys $RESULT/checkpoint_last5.gen.sys \
--ref $RESULT/checkpoint_last5.gen.ref
```

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
DATA=examples/translation/wmt14_en_de
BIN=data-bin/wmt14_en_de
python fairseq_cli/preprocess.py \
--source-lang en --target-lang de \
--trainpref $DATA/train --validpref $DATA/valid --testpref $DATA/test \
--destdir $BIN --thresholdtgt 0 --thresholdsrc 0 \
--workers 20 --joined_dictionary

# Train the model w/o cutoff
RESULT=results/transformer_base_wmt14_en2de
mkdir -p $RESULT
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
fairseq_cli/train.py $BIN \
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
--keep-last-epochs 20 \
--log-format tqdm \
--log-interval 100 \
--save-dir $RESULT \
--seed 97926458

# Train the model w/ cutoff
RESULT=results/transformer_base_wmt14_en2de_cutoff
mkdir -p $RESULT
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
fairseq_cli/train.py $BIN \
--arch transformer_wmt_en_de \
--augmentation \
--augmentation_schema cut_off \
--augmentation_masking_schema word \
--augmentation_masking_probability 0.05 \
--augmentation_replacing_schema mask \
--share-all-embeddings \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--adam-eps 1e-9 \
--clip-norm 0.0 \
--weight-decay 0.0 \
--criterion label_smoothed_cross_entropy_with_regularization \
--regularization_weight 5.0 \
--label-smoothing 0.1 \
--max-tokens 4096 \
--dropout 0.1 \
--lr-scheduler inverse_sqrt \
--lr 7e-4 \
--warmup-updates 8000 \
--max-epoch 100 \
--update-freq 1 \
--distributed-world-size 4 \
--ddp-backend=c10d \
--keep-last-epochs 20 \
--log-format tqdm \
--log-interval 100 \
--save-dir $RESULT \
--seed 5998856

# Evaluate
RESULT=results/transformer_base_wmt14_en2de
# RESULT=results/transformer_base_wmt14_en2de_cutoff
# average last 5 checkpoints
python scripts/average_checkpoints.py \
--inputs $RESULT \
--num-epoch-checkpoints 5 \
--output $RESULT/checkpoint_last5.pt
# generate results & quick evaluate
LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py $BIN \
--path $RESULT/checkpoint_last5.pt \
--beam 5 --remove-bpe --lenpen 0.0 >> $RESULT/checkpoint_last5.gen
# compound split & re-run evaluate
bash compound_split_bleu.sh $RESULT/checkpoint_last5.gen
LC_ALL=C.UTF-8 python fairseq_cli/score.py \
--sys $RESULT/checkpoint_last5.gen.sys \
--ref $RESULT/checkpoint_last5.gen.ref
```
