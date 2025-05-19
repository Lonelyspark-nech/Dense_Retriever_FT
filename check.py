import numpy as np
import faiss
import json
from pathlib import Path

# ==== 修改为你的路径 ====
corpus_path = "datasets_sample/corpus.tsv"
vecs_path = "cache/corpus_vecs.npy"
index_path = "cache/corpus_faiss.index"
query_jsonl = "datasets_sample/dev_queries_input.jsonl"
query_vecs_path = "cache/val_query_epoch1_vecs.npy"
# ========================

print("=== 检查 corpus 向量与 index_to_docid 映射 ===")
# 加载 corpus
corpus_ids, corpus_texts = [], []
with open(corpus_path, encoding="utf-8") as f:
    for line in f:
        docid, text = line.strip().split("\t")
        corpus_ids.append(docid)
        corpus_texts.append(text)

# 加载向量
vecs = np.load(vecs_path)
assert len(corpus_ids) == vecs.shape[0], "[❌] corpus_vecs 与 ID 数量不一致"
print(f"[✅] corpus 向量数量: {vecs.shape[0]}, 维度: {vecs.shape[1]}")

# 构建 FAISS 索引映射
index_to_docid = {i: docid for i, docid in enumerate(corpus_ids)}

# 加载 FAISS index
assert Path(index_path).exists(), f"未找到索引文件: {index_path}"
index = faiss.read_index(str(index_path))
print(f"[✅] 加载索引完成，索引维度: {index.d}")
assert vecs.shape[1] == index.d, "[❌] 向量维度与索引维度不匹配"

# 检查前几项
print("\n[🧪] index_to_docid 映射前5条:")
for i in range(5):
    print(f" - FAISS ID {i} → docid: {index_to_docid[i]} → text[:30]: {corpus_texts[i][:30]}")

# 随机 query 检索测试
print("\n[🔍] FAISS 检索映射 sanity check")
faiss.normalize_L2(vecs)
query = vecs[0:1]
D, I = index.search(query, 5)
print("Top-5 FAISS IDs:", I[0])
print("Top-5 Doc IDs:", [index_to_docid[idx] for idx in I[0]])

# =========== 验证 query 向量和 query_ids 顺序 ==============
print("\n=== 检查 query 向量与 query_id 顺序是否一致 ===")
query_ids = []
with open(query_jsonl, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        query_ids.append(obj["query_id"])

query_vecs = np.load(query_vecs_path)
assert len(query_ids) == query_vecs.shape[0], "[❌] query 向量数量与 query ID 数量不一致"
print(f"[✅] query 向量数量: {query_vecs.shape[0]}, 维度: {query_vecs.shape[1]}")

print("\n[🧪] query_id 前5个:")
for i in range(5):
    print(f" - query_id: {query_ids[i]}, 向量前3维: {query_vecs[i][:3]}")
