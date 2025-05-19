#!/bin/bash

python train_triplet_static.py \
  --triplet_path ./datasets_sample/train_triplets.jsonl \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --save_dir outputs/repllama-lora \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-5 \
  --use_lora \
  --validate_every_epoch \
  --corpus_path ./datasets_sample/corpus.tsv \
  --dev_queries_path ./datasets_sample/dev_queries_input.jsonl \
  --dev_qrels_path ./datasets_sample/dev_qrels.tsv \
  --gradient_accumulation_steps 2 