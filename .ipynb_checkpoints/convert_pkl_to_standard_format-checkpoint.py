import argparse
import json
import pickle
from pathlib import Path

def load_small_corpus_ids(corpus_pkl):
    with open(corpus_pkl, "rb") as f:
        corpus = pickle.load(f)
    if isinstance(corpus, dict):
        return set(corpus.keys())
    elif isinstance(corpus, list):
        ids = set()
        for item in corpus:
            if isinstance(item, dict):
                ids.add(item.get("doc_id"))
            else:
                ids.add(getattr(item, "doc_id", None))
        return set(i for i in ids if i is not None)
    else:
        raise TypeError(f"Unsupported corpus format: {type(corpus)}")

def convert_corpus(corpus_pkl, output_path):
    with open(corpus_pkl, "rb") as f, open(output_path, "w", encoding="utf-8") as out:
        corpus = pickle.load(f)
        if isinstance(corpus, dict):
            for doc_id, text in corpus.items():
                out.write(f"{doc_id}\tpassage: {text.strip()}\n")
        elif isinstance(corpus, list):
            for item in corpus:
                if isinstance(item, dict):
                    doc_id = item.get("doc_id")
                    text = item.get("text", "").strip()
                else:
                    doc_id = getattr(item, "doc_id", None)
                    text = getattr(item, "text", "").strip()
                if doc_id:
                    out.write(f"{doc_id}\tpassage: {text}\n")
        else:
            raise TypeError(f"Unsupported corpus format: {type(corpus)}")
    print(f"[✅] Saved corpus to {output_path}")

def convert_queries_pkl(pkl_path, output_path):
    with open(pkl_path, "rb") as f, open(output_path, "w", encoding="utf-8") as out:
        queries = pickle.load(f)

        if isinstance(queries, dict):
            for qid, text in queries.items():
                obj = {"query_id": qid, "query": "query: " + text.strip()}
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")

        elif isinstance(queries, list):
            for item in queries:
                if isinstance(item, dict):
                    qid = item.get("query_id")
                    text = item.get("query") or item.get("text", "")
                else:
                    qid = getattr(item, "query_id", None)
                    text = getattr(item, "query", "") or getattr(item, "text", "")
                if qid:
                    obj = {"query_id": qid, "query": "query: " + text.strip()}
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        else:
            raise TypeError(f"Unsupported queries format: {type(queries)}")

    print(f"[✅] Saved queries to {output_path}")


    print(f"[✅] Saved queries to {output_path}")

def convert_qrels_pkl(pkl_path, output_path):
    with open(pkl_path, "rb") as f, open(output_path, "w", encoding="utf-8") as out:
        qrels = pickle.load(f)

        if isinstance(qrels, dict):
            for qid, doc_ids in qrels.items():
                for doc_id in doc_ids:
                    out.write(f"{qid}\t0\t{doc_id}\t1\n")

        elif isinstance(qrels, list):
            for item in qrels:
                if isinstance(item, dict):
                    qid = item.get("query_id")
                    docid = item.get("doc_id")
                else:
                    qid = getattr(item, "query_id", None)
                    docid = getattr(item, "doc_id", None)
                if qid and docid:
                    out.write(f"{qid}\t0\t{docid}\t1\n")
        else:
            raise TypeError(f"Unsupported qrels format: {type(qrels)}")

    print(f"[✅] Saved qrels to {output_path}")


from tqdm import tqdm

def filter_triplets(triplet_path, output_path, small_corpus_ids, min_negatives=3, max_negatives=10):
    with open(triplet_path, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    kept = 0
    with open(triplet_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total_lines, desc="Filtering triplets"):
            obj = json.loads(line)
            pos_id = obj["positive_id"]
            neg_ids = obj["negative_ids"]
            valid_negs = [nid for nid in neg_ids if nid in small_corpus_ids]
            if pos_id in small_corpus_ids and len(valid_negs) >= min_negatives:
                obj["negative_ids"] = valid_negs[:max_negatives]
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[🎯] Filtered {kept} triplets from {total_lines} total based on corpus.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_pkl", type=str, required=True, help="Path to small corpus.pkl")
    parser.add_argument("--triplet_path", type=str, required=True, help="Path to original triplets JSONL")
    parser.add_argument("--output_triplets", type=str, required=True, help="Filtered triplets output JSONL path")
    parser.add_argument("--dev_queries_pkl", type=str, help="Optional: path to dev queries .pkl")
    parser.add_argument("--dev_qrels_pkl", type=str, help="Optional: path to dev qrels .pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save converted files")
    parser.add_argument("--min_neg", type=int, default=3, help="Minimum number of valid negatives required")
    parser.add_argument("--max_neg", type=int, default=10, help="Maximum number of negatives to keep")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_corpus(args.corpus_pkl, output_dir / "corpus.tsv")
    small_corpus_ids = load_small_corpus_ids(args.corpus_pkl)
    filter_triplets(args.triplet_path, args.output_triplets, small_corpus_ids,
                    min_negatives=args.min_neg, max_negatives=args.max_neg)

    if args.dev_queries_pkl:
        convert_queries_pkl(args.dev_queries_pkl, output_dir / "dev_queries_input.jsonl")
    if args.dev_qrels_pkl:
        convert_qrels_pkl(args.dev_qrels_pkl, output_dir / "dev_qrels.tsv")

if __name__ == "__main__":
    main()
