#!/bin/bash

python convert_pkl_to_standard_format.py \
  --corpus_pkl ./ir_cache_top10/msmarco_docs.pkl \
  --dev_queries_pkl ./ir_cache_top10/msmarco_queries.pkl \
  --dev_qrels_pkl  ./ir_cache_top10/msmarco_qrels.pkl \
  --triplet_path ./datasets/train_triplets.jsonl \
  --output_dir ./dataset_sample \
  --min_neg 3 \
  --max_neg 1000 \
  --output_triplets ./dataset_sample/train_triplets_small.jsonl




