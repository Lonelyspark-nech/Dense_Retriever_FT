import random
import json
from torch.utils.data import Dataset
import csv

class TripletDataset(Dataset):
    """
    接收格式为 List[Dict]，每项包含 query, positive, negatives（列表）的三元组数据集。
    """
    def __init__(self, triplets, num_negatives=3):
        self.triplets = triplets
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]
        query = item['query']
        positive = item['positive']
        negatives = item['negatives']

        # 采样固定数量的负例（随机采样）
        if len(negatives) >= self.num_negatives:
            sampled_negs = random.sample(negatives, self.num_negatives)
        else:
            sampled_negs = negatives + random.choices(negatives, k=self.num_negatives - len(negatives))

        return {
            'query': query,
            'positive': positive,
            'negatives': sampled_negs
        }

def load_triplets_with_padding(filepath, min_negatives=1):
    triplets = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            # 兼容新版结构：从 negative_ids 获取
            if "negative_ids" in data:
                query = data["query"]
                positive = data["positive"]
                negatives = data.get("negative_ids", [])
            else:
                # 保留旧结构兼容性
                query = data["query"]
                positive = data["positive"]
                negatives = data.get("negatives", [])

            # 忽略负例不足的样本
            if len(negatives) < min_negatives:
                continue

            triplets.append({
                "query": query,
                "positive": positive,
                "negatives": negatives
            })
    print(f"[✓] Loaded {len(triplets)} triplets from {filepath}")
    return triplets

    
def load_retrieval_dataset(query_path, qrels_path, corpus_path):
    """
    加载验证集或测试集所需的数据结构，包含 query, corpus, qrels 映射字典。
    返回结构:
        {
            "queries": Dict[query_id -> query_text],
            "corpus": Dict[doc_id -> passage_text],
            "qrels": Dict[query_id -> Set[relevant_doc_ids]]
        }
    """

    queries = {}
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["query_id"]] = obj["query"]

    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for row in tsv_reader:
            if len(row) < 2:
                continue
            doc_id, text = row[0], row[1]
            corpus[doc_id] = text

    qrels = {}
    with open(qrels_path, 'r', encoding='utf-8') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for row in tsv_reader:
            if len(row) < 3:
                continue
            query_id, doc_id = row[0], row[2]
            if query_id not in qrels:
                qrels[query_id] = set()
            qrels[query_id].add(doc_id)

    return {
        "queries": queries,
        "corpus": corpus,
        "qrels": qrels
    }