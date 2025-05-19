import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import os
import time
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
import numpy as np
import faiss

from model_utils import encode_with_eos, encode_and_cache, build_faiss_index, build_model_and_tokenizer
from dataset_utils import TripletDataset, load_triplets_with_padding

from collections import defaultdict

import argparse

from torch.cuda.amp import autocast
from peft import get_peft_model_state_dict

parser = argparse.ArgumentParser()
parser.add_argument("--triplet_path", type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--validate_every_epoch", action="store_true", help="是否在每个 epoch 后进行验证")
parser.add_argument("--corpus_path", type=str, help="验证集对应的 corpus TSV 文件路径")
parser.add_argument("--dev_queries_path", type=str, help="验证集的 query 文件路径 (.jsonl)")
parser.add_argument("--dev_qrels_path", type=str, help="验证集的 qrels 文件路径 (.tsv)")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="多少个 step 累积一次梯度更新（显存不足时有效扩大 batch）")
args = parser.parse_args()


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def cosine_similarity(a, b):
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.matmul(a_norm, b_norm.T)

def check_model_config(model):
    expected_dim = 4096
    actual_dim = getattr(model.config, "hidden_size", None)
    print(f"\n[🧪] Model Hidden Size Check:")
    print(f"Expected: {expected_dim}, Actual: {actual_dim}")
    if actual_dim != expected_dim:
        print(f"[❌] Mismatch detected! Please check config.json.")
    else:
        print(f"[✅] Hidden size matches expected LLaMA2-7B (4096).")

def collate_fn(batch):
    return {
        "query": [item["query"] for item in batch],
        "positive": [item["positive"] for item in batch],
        "negatives": [item["negatives"] for item in batch]
    }

def train_one_epoch(model, tokenizer, dataloader, optimizer, scheduler, device, accumulation_steps=1):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        queries = [f"query: {q}" for q in batch['query']]
        positives = [f"passage: {p}" for p in batch['positive']]
        negatives = [[f"passage: {n}" for n in neg_list] for neg_list in batch['negatives']]
    
        B = len(queries)
    
        negatives_flat = [n for negs in negatives for n in negs]

        all_texts = (
            [f"query: {q}" for q in batch["query"]] +
            [f"passage: {p}" for p in batch["positive"]] +
            [f"passage: {n}" for n in negatives_flat]
        )
        
        all_vecs = encode_with_eos(
            model, tokenizer, all_texts,
            device=device,
            dtype=model.dtype,
            max_length=512,
            no_grad=False  # 因为你正在训练
        )
        
        # ✂️ 拆分向量
        B = len(batch["query"])
        N = len(negatives[0])
        
        q_vecs, p_vecs, n_vecs_flat = all_vecs[:B], all_vecs[B:B*2], all_vecs[B*2:]
        
        # 🚫 禁止从 numpy 转换，避免 requires_grad=False
        assert isinstance(q_vecs, torch.Tensor), "[❌] q_vecs 不是 Tensor，请检查 encode_with_eos() 是否返回了 numpy！"
        
        # ✅ 保证张量位于正确设备上
        q_vecs = q_vecs.to(device)
        p_vecs = p_vecs.to(device)
        n_vecs_flat = n_vecs_flat.to(device)
        
        # ✅ reshape negatives
        D = q_vecs.shape[-1]
        n_vecs = n_vecs_flat.reshape(B, N, D)

        # ✅ 添加断言确保可反向传播
        assert q_vecs.requires_grad, "[❌] q_vecs 不可反向传播，可能 encode_with_eos() 返回了 numpy 或 detach() 了"
        assert p_vecs.requires_grad, "[❌] p_vecs 不可反向传播"
        assert n_vecs.requires_grad, "[❌] n_vecs 不可反向传播"

        # 归一化
        q_vecs = torch.nn.functional.normalize(q_vecs, p=2, dim=1)
        p_vecs = torch.nn.functional.normalize(p_vecs, p=2, dim=1)
        n_vecs = torch.nn.functional.normalize(n_vecs, p=2, dim=2)

        if step < 2:
            def print_stats(name, vec):
                print(f"[🔬] {name}:")
                print(f" - mean: {vec.mean().item():.4f}")
                print(f" - std:  {vec.std().item():.4f}")
                print(f" - norm: {vec.norm(dim=-1).mean().item():.4f}")
                print(f" - first 5 dims: {vec[0][:5].tolist()}")
        
            print_stats("q_vec", q_vecs)
            print_stats("p_vec", p_vecs)
            print_stats("n_vec[0]", n_vecs[0])
        
        all_doc_vecs = torch.cat([p_vecs.unsqueeze(1), n_vecs], dim=1)
        sim_scores = torch.einsum("bd,bnd->bn", q_vecs, all_doc_vecs)
        labels = torch.zeros(B, dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(sim_scores, labels)

        if step < 2:
            print("[🧪] Sample texts:")
            print("  query:   ", batch['query'][0])
            print("  positive:", batch['positive'][0])
            print("  negatives:", batch['negatives'][0])
        
            cos_qp = torch.cosine_similarity(q_vecs, p_vecs).mean().item()
            cos_qn = [torch.cosine_similarity(q_vecs[0:1], n.unsqueeze(0)).item() for n in n_vecs[0]]
        
            print(f"[🧪] Cosine(q, p): {cos_qp:.4f}")
            print(f"[🧪] Cosine(q, negs): {cos_qn}")
            print(f"[🧪] Sim scores row: {sim_scores[0].tolist()}")

        if step < 2 and (sim_scores < 0).all():
            print("[❗] All sim_scores < 0! Likely issue in vector direction or training collapse.")

        if step < 3:
            print("query:", batch["query"][0])
            print("positive:", batch["positive"][0])
            print("negatives:", batch["negatives"][0])
            print("sim_scores[0]:", sim_scores[0].tolist())
            print("q vs p cosine:", torch.cosine_similarity(q_vecs, p_vecs).mean().item())
            print(f"[❌] q vs neg cosine: {[torch.dot(q_vecs[0], n).item() for n in n_vecs[0]]}")
    
        loss = loss / accumulation_steps  # ✅ 累积步长缩放
        loss.backward()

        progress_bar.set_postfix({
            "sim_mean": f"{sim_scores.mean().item():.4f}",
            "loss": f"{loss.item():.4f}"
        })

    
        # ✅ 每 accumulation_steps 步更新一次参数
        if (step + 1) % accumulation_steps == 0 or (step + 1 == len(dataloader)):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
        total_loss += loss.item()

    return total_loss / len(dataloader)


def compute_mrr(retrieved_list, relevant_set):
    for rank, docid in enumerate(retrieved_list[:10], start=1):
        if docid in relevant_set:
            return 1.0 / rank
    return 0.0

def run_validation(index, query_vecs, qrels, top_k=100, index_to_docid=None):
    print("[🔍] Running validation retrieval...")

    assert query_vecs.shape[1] == index.d, (
        f"[❌] Dim mismatch: query_vecs={query_vecs.shape[1]}, FAISS index={index.d}"
        f"\n→ 可能 query 向量未正确降维，或使用了不同的 PCA"
    )
    faiss.normalize_L2(query_vecs)
    D, I = index.search(query_vecs, top_k)

    recalls, mrrs = [], []
    no_result = 0
    for i, qid in enumerate(qrels):
        retrieved = [index_to_docid[idx] for idx in I[i] if idx != -1]
        if len(retrieved) == 0:
            no_result += 1
            recalls.append(0)
            mrrs.append(0.0)
            continue
        rels = set(map(str, qrels[qid]))

        if i < 5:
            print(f"\n[🔍] Query ID: {qid}")
            print(f"[🔍] Relevant doc_ids in qrels: {list(rels)}")
            print(f"[🔍] Top-{top_k} Retrieved IDs: {retrieved[:10]}")
            hits = rels & set(retrieved)
            print(f"[✅] Hit: {bool(hits)}, Match(es): {list(hits)}")

        recalls.append(int(bool(rels & set(retrieved[:top_k]))))
        mrrs.append(compute_mrr(retrieved, rels))

    recall_at_k = sum(recalls) / len(recalls)
    mrr_at_10 = sum(mrrs) / len(mrrs)
    print(f"[📊] Recall@{top_k}: {recall_at_k:.4f}, MRR@10: {mrr_at_10:.4f}")
    print(f"[📉] Queries with no retrieved results: {no_result} / {len(qrels)}")
    return recall_at_k

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num = args.epochs
    cache_dir = "./cache"
    checkpoint_dic_path = "./checkpoints/repllama"
    os.makedirs(checkpoint_dic_path, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer()
    model.to(device)
    check_model_config(model)

    # 冻结非 LoRA 参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora" not in name:
            param.requires_grad = False
    lora_params = get_peft_model_state_dict(model)
    print(f"[🔍] Loaded LoRA adapter param count: {len(lora_params)}")
    print(f"[🔍] Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.print_trainable_parameters()

    # Load triplet training data
    triplets = load_triplets_with_padding(args.triplet_path, min_negatives=3)
    dataset = TripletDataset(triplets, num_negatives=3)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    total_steps = len(dataloader) * epoch_num
    warmup_steps = int(0.1 * total_steps)  # 可调，比如 10%
    scheduler = get_scheduler(
        "linear", optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Encode validation corpus and queries (only once)
    corpus_ids, corpus_texts = [], []
    with open(args.corpus_path, encoding="utf-8") as f:
        for line in f:
            doc_id, text = line.strip().split("\t")
            corpus_ids.append(doc_id)
            corpus_texts.append(text)
    query_ids, query_texts = [], []
    with open(args.dev_queries_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            query_ids.append(obj["query_id"])
            query_texts.append(obj["query"])

            
    qrels = defaultdict(list)
    with open(args.dev_qrels_path, encoding="utf-8") as f:
        for line in f:
            qid, _, docid, _ = line.strip().split("\t")
            qrels[qid].append(docid)
    corpus_vecs = encode_and_cache("corpus", corpus_texts, corpus_ids, model, tokenizer, cache_dir, pca_path=512)
    
    index_path = Path(cache_dir) / "corpus_faiss.index"
    index = build_faiss_index(corpus_vecs, index_path=index_path)

    index_to_docid = {i: docid for i, docid in enumerate(corpus_ids)}

    for epoch in range(epoch_num):
        print(f"\n[Epoch {epoch+1}/{epoch_num}]")
        avg_loss = train_one_epoch(model, tokenizer, dataloader, optimizer, scheduler, device,
                           accumulation_steps=args.gradient_accumulation_steps)
        print(f"Avg Loss: {avg_loss:.4f}")

        query_vecs = encode_and_cache(f"val_query_epoch{epoch+1}", query_texts, query_ids, model, tokenizer, cache_dir, pca_path=512)
        run_validation(index, query_vecs, qrels, top_k=100, index_to_docid=index_to_docid)

        out_dir = os.path.join(checkpoint_dic_path, f"epoch{epoch+1}")
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"[✓] Saved checkpoint to {out_dir}")

if __name__ == "__main__":
    main()
