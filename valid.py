import argparse
import json
from pathlib import Path
import numpy as np
import faiss
import torch
from collections import defaultdict
import logging

from model_utils import build_model_and_tokenizer, encode_and_cache


def compute_mrr(retrieved, relevant, top_k=10):
    for i, docid in enumerate(retrieved[:top_k], start=1):
        if docid in relevant:
            return 1.0 / i
    return 0.0


def run_validation(index, corpus_ids, query_vecs, qrels, top_k=100):
    D, I = index.search(query_vecs, top_k)
    recalls, mrrs = [], []
    for i, qid in enumerate(qrels.keys()):
        retrieved = [corpus_ids[idx] for idx in I[i] if idx != -1]
        relevant = set(qrels[qid])
        recalls.append(int(bool(relevant & set(retrieved[:top_k]))))
        mrrs.append(compute_mrr(retrieved, relevant, top_k=top_k))
    recall_at_k = np.mean(recalls)
    mrr_at_k = np.mean(mrrs)
    return recall_at_k, mrr_at_k


def main():
    parser = argparse.ArgumentParser(description="Validate triplet model with FAISS retrieval metrics.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to saved model/tokenizer checkpoint")
    parser.add_argument("--corpus_path", type=str, required=True,
                        help="TSV file of corpus: doc_id<tab>text per line")
    parser.add_argument("--dev_queries_path", type=str, required=True,
                        help="JSONL file of queries: {\"query_id\": ..., \"query\": ...}\n")
    parser.add_argument("--dev_qrels_path", type=str, required=True,
                        help="TSV file of qrels: query_id<tab>...<tab>doc_id<tab>... per line")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory for vector storage")
    parser.add_argument("--pca_path", type=str, default=None,
                        help="PCA path or None to disable PCA")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for encoding")
    parser.add_argument("--log_file", type=str, default="validate.log")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()])

    device = torch.device(args.device)
    # Load model & tokenizer
    model, tokenizer = build_model_and_tokenizer(lora_path=args.checkpoint_dir)
    model.to(device)
    model.eval()

    # Load corpus
    corpus_ids, corpus_texts = [], []
    with open(args.corpus_path, encoding="utf-8") as f:
        for line in f:
            docid, text = line.strip().split("\t", 1)
            corpus_ids.append(docid)
            corpus_texts.append(text)
    logging.info(f"Loaded {len(corpus_ids)} corpus documents.")

    # Encode corpus
    corpus_vecs = encode_and_cache(
        name="val_corpus",
        texts=corpus_texts,
        ids=corpus_ids,
        model=model,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        pca_path=args.pca_path,
        normalize=True,
        prefix="passage: ",
        batch_size=args.batch_size,
        rebuild=True
    )
    logging.info(f"Encoded corpus vectors shape: {corpus_vecs.shape}")

    # Build FAISS index
    index = faiss.IndexFlatIP(corpus_vecs.shape[1])
    index.add(corpus_vecs.astype(np.float32))
    logging.info("Built FAISS IndexFlatIP for corpus.")

    # Load queries
    query_ids, query_texts = [], []
    with open(args.dev_queries_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            query_ids.append(obj["query_id"])
            query_texts.append(obj["query"])
    logging.info(f"Loaded {len(query_ids)} queries.")

    # Load qrels
    qrels = defaultdict(list)
    with open(args.dev_qrels_path, encoding="utf-8") as f:
        for line in f:
            qid, _, docid, _ = line.strip().split("\t")
            qrels[qid].append(docid)
    logging.info(f"Loaded qrels for {len(qrels)} queries.")

    # Encode queries
    query_vecs = encode_and_cache(
        name="val_queries",
        texts=query_texts,
        ids=query_ids,
        model=model,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        pca_path=args.pca_path,
        normalize=True,
        prefix="query: ",
        batch_size=args.batch_size,
        rebuild=True
    )
    logging.info(f"Encoded query vectors shape: {query_vecs.shape}")

    # Run validation
    recall, mrr = run_validation(index, corpus_ids, query_vecs, qrels, top_k=100)
    logging.info(f"Validation Results → Recall@100: {recall:.4f}, MRR@10: {mrr:.4f}")
    print(f"Recall@100: {recall:.4f}, MRR@10: {mrr:.4f}")


if __name__ == "__main__":
    main()
