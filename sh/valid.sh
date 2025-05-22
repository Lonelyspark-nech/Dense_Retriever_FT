#!/bin/bash

python valid.py \
  --checkpoint_dir ./checkpoints/epoch3 \
  --corpus_path ./data/corpus.tsv \
  --dev_queries_path ./data/dev_queries.jsonl \
  --dev_qrels_path ./data/dev_qrels.tsv \
  --cache_dir ./cache \
  --pca_path none