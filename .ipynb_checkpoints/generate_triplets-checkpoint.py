import ir_datasets
import random
import json
from collections import defaultdict
from tqdm import tqdm


def generate_triplets_ir_dataset(
    dataset_name="msmarco-passage/train",
    top_k=100,
    max_queries=10000,
    output_path="triplets_kaggle.jsonl"
):
    """
    从 ir_datasets 加载 MS MARCO 训练集，构造静态三元组 (query, pos, negs)，存储为 jsonl。
    """
    dataset = ir_datasets.load(dataset_name)
    print(f"[✓] Loaded dataset: {dataset_name}")

    qrels = defaultdict(set)  # query_id -> set(doc_id)
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels[qrel.query_id].add(qrel.doc_id)

    # 准备查询、文档索引
    query_lookup = {q.query_id: q.text for q in dataset.queries_iter()}
    doc_lookup = {d.doc_id: d.text for d in tqdm(dataset.docs_iter(), desc="Indexing docs")}

    scoreddocs = list(dataset.scoreddocs_iter())
    topk_by_qid = defaultdict(list)
    for sdoc in scoreddocs:
        if len(topk_by_qid[sdoc.query_id]) < top_k:
            topk_by_qid[sdoc.query_id].append(sdoc.doc_id)

    triplets = []
    for qid, docids in tqdm(topk_by_qid.items(), desc="Building triplets"):
        if qid not in qrels or len(qrels[qid]) == 0:
            continue

        pos_ids = list(qrels[qid])
        neg_ids = [did for did in docids if did not in qrels[qid]]

        if not neg_ids:
            continue

        pos_id = random.choice(pos_ids)
        negs = random.sample(neg_ids, k=min(5, len(neg_ids)))

        triplets.append({
            "query": query_lookup[qid],
            "positive": doc_lookup.get(pos_id, ""),
            "negatives": [doc_lookup.get(nid, "") for nid in negs]
        })

        if len(triplets) >= max_queries:
            break

    with open(output_path, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            json.dump(triplet, f, ensure_ascii=False)
            f.write("\n")

    print(f"[✓] Wrote {len(triplets)} triplets to {output_path}")