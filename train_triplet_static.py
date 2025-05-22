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
import csv
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from model_utils import encode_with_eos, encode_and_cache, build_faiss_index, build_model_and_tokenizer
from dataset_utils import TripletDataset, load_triplets_with_padding

from collections import defaultdict

import argparse

from torch.cuda.amp import autocast
from peft import get_peft_model_state_dict

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler(enabled=False)


def parse_pca(val):
    if val.lower() == "none":
        return None
    try:
        return int(val)
    except ValueError:
        return str(val)

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
parser.add_argument("--pca_path", type=parse_pca, default=512, help="PCA 维度（int），或模型路径（str），或 None 禁用")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Path to checkpoint to resume training from")
parser.add_argument("--debug_only", action="store_true",
                    help="Skip training and run a single validation")
parser.add_argument("--log_file", type=str, default="training.log")
args = parser.parse_args()


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

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
        print(f"Mismatch detected! Please check config.json.")
    else:
        print(f"Hidden size matches expected LLaMA2-7B (4096).")

def collate_fn(batch):
    return {
        "query": [item["query"] for item in batch],
        "positive": [item["positive"] for item in batch],
        "negatives": [item["negatives"] for item in batch]
    }

def train_one_epoch(epoch,model, tokenizer, dataloader, optimizer, scheduler, device, accumulation_steps=1):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):   # 前向都在半精度下跑
            queries = [f"query: {q}" for q in batch["query"]]
            positives = [f"passage: {p}" for p in batch["positive"]]
            negatives = batch["negatives"] 
            negatives_flat = [f"passage: {n}" for negs in negatives for n in negs]

            all_texts = queries + positives + negatives_flat
            
            all_vecs = encode_with_eos(
                model, tokenizer, all_texts,  
                device=device,
                # dtype=torch.float32,
                max_length=512,
                batch_size=len(all_texts),
                no_grad=False,   # 因为你正在训练
                prefix=None
            )
            # —— 强制转为 float32，保证与 optimizer state 一致
            # all_vecs = all_vecs.to(torch.float32)
            
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

            # print(f"[DEBUG] model.dtype = {model.dtype}")
            # print(f"[DEBUG] q_vecs.requires_grad = {getattr(q_vecs, 'requires_grad', 'N/A')}")
            # ✅ 添加断言确保可反向传播
            assert q_vecs.requires_grad, "[❌] q_vecs 不可反向传播，可能 encode_with_eos() 返回了 numpy 或 detach() 了"
            assert p_vecs.requires_grad, "[❌] p_vecs 不可反向传播"
            assert n_vecs.requires_grad, "[❌] n_vecs 不可反向传播"

            # 归一化
            q_vecs = torch.nn.functional.normalize(q_vecs, p=2, dim=1)
            p_vecs = torch.nn.functional.normalize(p_vecs, p=2, dim=1)
            n_vecs = torch.nn.functional.normalize(n_vecs, p=2, dim=2)
            
            all_doc_vecs = torch.cat([p_vecs.unsqueeze(1), n_vecs], dim=1)

            temperature = 0.1  # 可以尝试不同温度：5.0、10.0、20.0等
            
            # similarity scores with temperature scaling
            sim_scores = torch.einsum("bd,bnd->bn", q_vecs, all_doc_vecs)
            labels = torch.zeros(B, dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(sim_scores, labels)

            loss = loss / accumulation_steps  # ✅ 累积步长缩放
            #loss.backward()

            # if step < 3:
            #     def print_vector_stats(name, vec):
            #         norms = vec.norm(dim=-1)
            #         print(f"[📏] {name} norm: min={norms.min().item():.4f}, max={norms.max().item():.4f}, mean={norms.mean().item():.4f}")
            #         print(f"[📉] {name} mean: {vec.mean().item():.6f}, std: {vec.std().item():.6f}")
            #     print_vector_stats("q_vecs", q_vecs)
            #     print_vector_stats("p_vecs", p_vecs)
            #     print_vector_stats("n_vecs", n_vecs)

            # if step < 2:
            #     print("[🧪] Sample texts:")
            #     print("  query:   ", batch['query'][0])
            #     print("  positive:", batch['positive'][0])
            #     print("  negatives:", batch['negatives'][0])
            
            #     cos_qp = torch.cosine_similarity(q_vecs, p_vecs).mean().item()
            #     cos_qn = [torch.cosine_similarity(q_vecs[0:1], n.unsqueeze(0)).item() for n in n_vecs[0]]
            
            #     print(f"[🧪] Cosine(q, p): {cos_qp:.4f}")
            #     print(f"[🧪] Cosine(q, negs): {cos_qn}")
            #     print(f"[🧪] Sim scores row: {sim_scores[0].tolist()}")

            # if step < 2 and (sim_scores < 0).all():
            #     print("[❗] All sim_scores < 0! Likely issue in vector direction or training collapse.")

            progress_bar.set_postfix({
                "sim_mean": f"{sim_scores.mean().item():.4f}",
                "loss": f"{loss.item():.4f}"
            })

    
        # ✅ 每 accumulation_steps 步更新一次参数
        if (step + 1) % accumulation_steps == 0 or (step + 1 == len(dataloader)):
            scaler.scale(loss).backward()      # 梯度被 scale 同时 float16→float32
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        # logging.info(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")

    return total_loss / len(dataloader)


def compute_mrr(retrieved_list, relevant_set):
    for rank, docid in enumerate(retrieved_list[:10], start=1):
        if docid in relevant_set:
            return 1.0 / rank
    return 0.0

def debug_alignment_sample(
    query_text: str,
    positive_docid: str,
    model,
    tokenizer,
    faiss_index,
    corpus_ids: list,
    docid_to_text: dict,
    doc_vectors: np.ndarray,
    pca_path: str,
    cache_dir: str,
    apply_norm: bool = True
):
    from model_utils import encode_with_eos, apply_pca
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    print(f"\n[🔎] [对齐检查] Query: {query_text}")
    print(f"[🔎] Positive docid: {positive_docid}")
    if positive_docid not in corpus_ids:
        print(f"[❌] 正例 docid {positive_docid} 不在 corpus_ids 中！")
        return
    doc_idx = corpus_ids.index(positive_docid)
    doc_vec = doc_vectors[doc_idx].reshape(1, -1)  # doc_vectors = corpus_vecs

    query_vec = encode_with_eos(model, tokenizer, [query_text], no_grad=True, prefix=None)
    query_vec = query_vec.astype(np.float32)

    if pca_path is not None:
        query_vec = apply_pca("debug_query", query_vec, pca_path, cache_dir)

    if apply_norm:
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        doc_vec = doc_vec / np.linalg.norm(doc_vec, axis=1, keepdims=True)

    sim = cosine_similarity(query_vec, doc_vec)[0][0]
    print(f"[📏] Cosine(query, positive_doc): {sim:.4f}")
    print(f"[🧾] 正例摘要: {docid_to_text[positive_docid][:300]}...")

    
def run_validation(epoch,metrics_csv,avg_loss,index, query_vecs, qrels, top_k=100, index_to_docid=None, query_ids=None, query_texts=None,model=None, tokenizer=None,
                   corpus_ids=None, docid_to_text=None,doc_vectors=None,
                   pca_path=None, cache_dir=None):
    print("[🔍] Running validation retrieval...")

    assert query_vecs.shape[1] == index.d, (
        f"[❌] Dim mismatch: query_vecs={query_vecs.shape[1]}, FAISS index={index.d}"
        f"\n→ 可能 query 向量未正确降维，或使用了不同的 PCA"
    )
    # faiss.normalize_L2(query_vecs)
    D, I = index.search(query_vecs, top_k)

    recalls, mrrs = [], []
    no_result = 0
    for i, qid in enumerate(query_ids):
        retrieved = [index_to_docid[idx] for idx in I[i] if idx != -1]

        # print(">>> 预测行 0 对应的 query_id:", query_ids[0])
        # print(">>> qrels 枚举的第一个 qid:", list(qrels.keys())[0])
        
        if len(retrieved) == 0:
            no_result += 1
            recalls.append(0)
            mrrs.append(0.0)
            continue
        rels = set(map(str, qrels[qid]))

        # if i < 5:
        #     print(f"\n[🔍] Query ID: {qid}")
        #     print(f"[🔍] Relevant doc_ids in qrels: {list(rels)}")
        #     print(f"[🔍] Top-{top_k} Retrieved IDs: {retrieved[:10]}")
        #     hits = rels & set(retrieved)
        #     print(f"[✅] Hit: {bool(hits)}, Match(es): {list(hits)}")
        # if i == 0:
        #     print("[🧪] Raw FAISS distances:", D[i][:10])
        #     print("[🧪] Top doc IDs:", I[i][:10])

        recalls.append(int(bool(rels & set(retrieved[:top_k]))))
        mrrs.append(compute_mrr(retrieved, rels))

    recall_at_k = sum(recalls) / len(recalls)
    mrr_at_10 = sum(mrrs) / len(mrrs)
    # print(f"Recall@{top_k}: {recall_at_k:.4f}, MRR@10: {mrr_at_10:.4f}")
    logging.info(f"Epoch {epoch+1} Val Recall@{top_k}: {recall_at_k:.4f}, MRR@10: {mrr_at_10:.4f}")
    # Append metrics
    with open(metrics_csv, 'a') as f:
        csv.writer(f).writerow([epoch, avg_loss, recall_at_k, mrr_at_10])

    # print(f"[📉] Queries with no retrieved results: {no_result} / {len(qrels)}")
    
    # 向量空间对齐检测：query vs corpus

    # corpus_sample = index.reconstruct_n(0, min(50, index.ntotal))  # numpy
    # q_sample = query_vecs[:min(50, len(query_vecs))]               # numpy
    # sim_matrix = sk_cosine_similarity(q_sample, corpus_sample)
    # sim_vals = sim_matrix.flatten()
    # print(f"[🔍] Cosine(q,p) sample similarity: mean={sim_vals.mean():.4f}, std={sim_vals.std():.4f}, min={sim_vals.min():.4f}, max={sim_vals.max():.4f}")


    # [🧪] 验证 query 和 positive doc 对齐情况（前 3 条）
    # print("\n[🧪] Running alignment check for a few queries...")
    # checked = 0
    # for i, qid in enumerate(qrels):
    #     if checked >= 3:
    #         break
    #     rels = qrels[qid]
    #     if len(rels) == 0:
    #         continue
    #     query_text = next((q for idx, q in zip(query_ids, query_texts) if idx == qid), None)
    #     if query_text is None:
    #         continue
            
    #     debug_alignment_sample(
    #         query_text="query: " + query_text,
    #         positive_docid=rels[0],
    #         model=model,
    #         tokenizer=tokenizer,
    #         faiss_index=index,
    #         corpus_ids=corpus_ids,
    #         docid_to_text=docid_to_text,
    #         doc_vectors=doc_vectors,
    #         pca_path=pca_path,
    #         cache_dir="./cache",
    #         apply_norm=True
    #     )
        
    
    return recall_at_k

def main():
    setup_logging(args.log_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num = args.epochs
    cache_dir = "./cache"
    checkpoint_dic_path = "./checkpoints/repllama"
    pca_path=args.pca_path
    os.makedirs(checkpoint_dic_path, exist_ok=True)

    # Load or initialize model & tokenizer
    if args.resume_from and args.resume_from.lower() != 'none':
        model, tokenizer = build_model_and_tokenizer(lora_path=args.resume_from)
        logging.info(f"Loaded model from {args.resume_from}")
    else:
        model, tokenizer = build_model_and_tokenizer()
    model.to(device)
    # check_model_config(model)

    # 冻结非 LoRA 参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora" not in name:
            param.requires_grad = False
    lora_params = get_peft_model_state_dict(model)
    print(f"Loaded LoRA adapter param count: {len(lora_params)}")
    print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.print_trainable_parameters()
    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         #p.data = p.data.float()

    #读取corpus
    corpus_ids, corpus_texts = [], []
    with open(args.corpus_path, encoding="utf-8") as f:
        for line in f:
            doc_id, text = line.strip().split("\t")
            corpus_ids.append(doc_id)
            corpus_texts.append(text)

    # Load triplet training data
    triplets = load_triplets_with_padding(args.triplet_path, corpus_ids,corpus_texts, min_negatives=3)
    dataset = TripletDataset(triplets, num_negatives=3)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    
    total_steps = len(dataloader) * epoch_num
    warmup_steps = int(0.1 * total_steps)  # 可调，比如 10%
    scheduler = get_scheduler(
        "linear", optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    start_epoch = 0
    if args.resume_from:
        # Resume optimizer/scheduler if checkpoint provided
        opt_path = Path(args.resume_from) / "optimizer.pt"
        sch_path = Path(args.resume_from) / "scheduler.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path))
            # 把所有 exp_avg / exp_avg_sq 转成参数同样的 dtype
            optimizer.load_state_dict(torch.load(opt_path))
            for group in optimizer.param_groups:
                group["foreach"] = False
                if "fused" in group:
                    group["fused"] = False
                for p in group["params"]:
                    state = optimizer.state[p]
                    if "exp_avg" in state:
                        state["exp_avg"]    = state["exp_avg"].float()
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"] = state["exp_avg_sq"].float()
            print({g['foreach'] for g in optimizer.param_groups},{getattr(g, 'fused', 'NA') for g in optimizer.param_groups})
            
        if sch_path.exists():
            scheduler.load_state_dict(torch.load(sch_path))
        try:
            last_epoch = int(Path(args.resume_from).name.replace("epoch", ""))
            start_epoch = last_epoch + 1
            logging.info(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            logging.warning(f"Failed to parse epoch number from resume path: {e}")
            
    # Encode validation corpus and queries
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

            
    # corpus_vecs = encode_and_cache("corpus", corpus_texts, corpus_ids, model, tokenizer, cache_dir, pca_path=512, normalize=True, prefix="passage: ")

    #  # ✅ 添加检查代码
    # print("[🔍] Validating corpus vector alignment...")
    # assert len(corpus_ids) == corpus_vecs.shape[0], (
    #     f"[❌] corpus_ids({len(corpus_ids)}) 和 corpus_vecs({corpus_vecs.shape[0]}) 数量不一致！"
    # )
    # print(f"[✅] corpus_ids 与 corpus_vecs 对齐：{len(corpus_ids)} vectors")
    
    
    # index_path = Path(cache_dir) / "corpus_faiss.index"
    
    # index = build_faiss_index(corpus_vecs, index_path=index_path)

    # import faiss
    # index = faiss.IndexFlatIP(corpus_vecs.shape[1])
    # index.add(corpus_vecs.astype(np.float32))
    # print("[🧪] Using FAISS FlatIP index for validation (no IVF, no training)")

    index_to_docid = {i: docid for i, docid in enumerate(corpus_ids)}

    # Prepare metrics file
    metrics_csv =Path("./metrics.csv")
    if not metrics_csv.exists():
        with open(metrics_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch","train_loss","val_recall","val_mrr"])

            
    # —— 用最新模型重编码语料 & 重建索引 —— 
    print("Re-encoding corpus with updated model...")
    corpus_vecs = encode_and_cache("corpus", corpus_texts, corpus_ids,
                                   model, tokenizer, cache_dir,
                                   pca_path=pca_path, normalize=True,
                                   prefix="passage: ",
                                   rebuild=False)
    print("Rebuilding FlatIP index...")
    index = faiss.IndexFlatIP(corpus_vecs.shape[1])
    index.add(corpus_vecs.astype(np.float32))

    

    for epoch in range(start_epoch, epoch_num):
        print(f"\n[Epoch {epoch+1}/{epoch_num}]")
        avg_loss = train_one_epoch(epoch+1, model, tokenizer, dataloader, optimizer, scheduler, device,
                           accumulation_steps=args.gradient_accumulation_steps)
        logging.info(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        # query_vecs = encode_and_cache(f"val_query_epoch{epoch+1}", query_texts, query_ids, model, tokenizer, cache_dir, pca_path=512, normalize=True,prefix="query: ")
        
        # print(f"[🧪] 验证向量: val_query_epoch{epoch+1}")
        # print(f" - 向量数量: {query_vecs.shape[0]}")
        # print(f" - 向量维度: {query_vecs.shape[1]}")
        # norms = np.linalg.norm(query_vecs, axis=1)
        # print(f" - 范数 min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
        # print(f" - 零向量数: {(norms < 1e-6).sum()} / {len(query_vecs)}")
        # print(f" - mean={query_vecs.mean():.6f}, std={query_vecs.std():.6f}")

        # run_validation(index, query_vecs, qrels, top_k=100, index_to_docid=index_to_docid, query_ids=query_ids, query_texts=query_texts,     model=model,tokenizer=tokenizer,corpus_ids=corpus_ids,docid_to_text=dict(zip(corpus_ids, corpus_texts)),doc_vectors=corpus_vecs,pca_path=512,cache_dir=cache_dir)
        

        
        # 再编码 query & 验证
        query_vecs = encode_and_cache(f"val_query_epoch{epoch+1}", query_texts, query_ids,
                                      model, tokenizer, cache_dir,
                                      pca_path=pca_path, normalize=True,
                                      prefix="query: ",
                                      rebuild=True)
        recall = run_validation(epoch, metrics_csv,avg_loss,index, query_vecs, qrels, top_k=100,
                                index_to_docid=index_to_docid,
                                query_ids=query_ids,
                                query_texts=query_texts,
                                model=model,
                                tokenizer=tokenizer,
                                corpus_ids=corpus_ids,
                                docid_to_text=dict(zip(corpus_ids, corpus_texts)),
                                doc_vectors=corpus_vecs,
                                pca_path=pca_path,
                                cache_dir=cache_dir)
        # Save checkpoint
        ckpt_dir = Path(args.save_dir) / f"epoch{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        logging.info(f"Saved checkpoint to {ckpt_dir}")

        # out_dir = os.path.join(checkpoint_dic_path, f"epoch{epoch+1}")
        # os.makedirs(out_dir, parents=True, exist_ok=True)
        # model.save_pretrained(out_dir)
        # tokenizer.save_pretrained(out_dir)
        # print(f"Saved checkpoint to {out_dir}")

if __name__ == "__main__":
    main()
