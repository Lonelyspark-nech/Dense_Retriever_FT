import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
from typing import List, Optional, Union
import contextlib
from sklearn.decomposition import PCA
from joblib import dump, load

from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import faiss
from transformers import AutoTokenizer, AutoModel

# 默认的 LoRA 配置，用于注入到 LLaMA 模型中
default_lora_config = LoraConfig(
    r=16,  # 低秩矩阵的秩 r，控制插入参数量（越大表示容量越大）
    lora_alpha=32,  # 缩放系数，放大低秩表示的影响
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,  # dropout 防止过拟合
    bias="none",  # 不修改 bias 项
    task_type=TaskType.FEATURE_EXTRACTION  # 不是文本生成任务，而是检索任务（提取向量）
)

def build_model_and_tokenizer(
    base_model_name="meta-llama/Llama-2-7b-hf",
    lora_config=default_lora_config,
    lora_path=None,
    use_bf16=True,
    device_map="auto"
):
    """
    加载基础 LLaMA 模型和 Tokenizer，如果指定了 LoRA adapter 路径则加载 adapter。
    返回 PEFT 模型和 tokenizer。
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA 模型无 pad token，使用 eos 填充

    model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map=device_map
    )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)  # 加载已有 adapter 权重
        print(f"[✓] Loaded LoRA adapter from {lora_path}")
    else:
        model = get_peft_model(model, lora_config)  # 注入新的 LoRA 层
        print("[✓] Injected new LoRA adapters")

    model.eval()  # 关闭 dropout、layernorm 等训练特有行为
    return model, tokenizer


def encode_with_eos(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 32,
    prefix: str = "",
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16,
    no_grad: bool = True,
) -> np.ndarray:
    if no_grad:
        model.eval()
    model.to(device=model.device, dtype=model.dtype)
    
    def _encode_batch(batch):
        assert all(len(t.strip()) > 0 for t in batch), "[❌] Found empty string in input batch!"
        inputs = tokenizer(
            [prefix + t + tokenizer.eos_token for t in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device=device) for k, v in inputs.items()}

        with torch.no_grad() if no_grad else contextlib.nullcontext():
            outputs = model(**inputs, return_dict=True)
            last_hidden_state = outputs.last_hidden_state

        lengths = inputs["attention_mask"].sum(dim=1) - 1
        pooled = torch.stack([
            last_hidden_state[i, l] if l < last_hidden_state.size(1) else torch.zeros_like(last_hidden_state[i, 0])
            for i, l in enumerate(lengths)
        ])

        if no_grad:
            return pooled.detach().cpu().to(torch.float32).numpy()
        else:
            return pooled  # 保留计算图

    all_vecs = []
    i = 0
    last_successful_bs = batch_size

    while i < len(texts):
        bs = last_successful_bs
        while bs >= 1:
            try:
                batch = texts[i:i+bs]
                vecs = _encode_batch(batch)
                all_vecs.append(vecs)
                i += bs
                last_successful_bs = bs
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    bs //= 2
                else:
                    raise e
        if bs < 1:
            raise RuntimeError(f"OOM at index {i}, even with batch_size=1")

    if no_grad:
        return np.concatenate(all_vecs, axis=0)
    else:
        return torch.cat(all_vecs, dim=0)




def apply_pca(name: str, vecs: np.ndarray, pca_path, cache_dir: Path) -> np.ndarray:
    """
    根据 pca_path 类型执行不同逻辑：
    - int: 使用共享 PCA（训练或加载 corpus_pca.pkl）
    - str: 加载指定 PCA 路径
    - None: 不降维
    """
    if isinstance(pca_path, int):
        shared_pca_path = Path(cache_dir) / "corpus_pca.pkl"
        skipped_flag_path = Path(cache_dir) / "corpus_pca_skipped.flag"

        if name == "corpus":
            if vecs.shape[0] < pca_path:
                print(f"[⚠️] Too few corpus samples ({vecs.shape[0]}) for PCA={pca_path}, skipping PCA.")
                skipped_flag_path.touch()
                return vecs
            else:
                print(f"[🔍] Training PCA on corpus to {pca_path} dims...")
                pca = PCA(n_components=pca_path)
                vecs = pca.fit_transform(vecs)
                dump(pca, shared_pca_path)
                print(f"[💾] Saved shared PCA model to {shared_pca_path}")
                if skipped_flag_path.exists():
                    skipped_flag_path.unlink()
                    print(f"[🧹] Removed old PCA skipped flag: {skipped_flag_path}")
                return vecs

        else:
            if not shared_pca_path.exists():
                if skipped_flag_path.exists():
                    raise ValueError(
                        "[❌] corpus PCA was skipped due to small sample size. "
                        "Validation cannot proceed with PCA.\n"
                        "→ Set pca_path=None or increase corpus size."
                    )
                else:
                    raise FileNotFoundError(
                        f"[❌] Shared PCA model {shared_pca_path} not found. "
                        f"Did you run corpus encoding first?"
                    )
            print(f"[🔁] Loading shared PCA model from {shared_pca_path}")
            pca = load(shared_pca_path)
            return pca.transform(vecs)

    elif isinstance(pca_path, str):
        pca_model_path = Path(pca_path)
        if not pca_model_path.exists():
            raise FileNotFoundError(f"[❌] PCA model not found at: {pca_model_path}")
        print(f"[🔁] Loading PCA model from {pca_model_path}")
        pca = load(pca_model_path)
        return pca.transform(vecs)

    elif pca_path is None:
        print("[ℹ️] PCA disabled: using original vector dimension.")
        return vecs

    else:
        raise TypeError(f"[❌] Invalid pca_path type: {type(pca_path)}")

def encode_and_cache(
    name: str,
    texts: List[str],
    ids: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    cache_dir: Union[str, Path],
    max_length: int = 512,
    batch_size: int = 128,
    prefix: str = "passage: ",
    pca_path: Optional[Union[int, str]] = None,
    normalize: bool = True,
    rebuild: bool = False
) -> np.ndarray:
    from sklearn.decomposition import PCA
    from joblib import dump, load

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{name}_vecs.npy"
    pca_model_path = cache_dir / f"{name}_pca.pkl"

    if cache_path.exists() and not rebuild:
        vecs = np.load(cache_path)
        print(f"[✅] Loaded cached vectors: {cache_path}")
    else:
        vecs = []
        model.to(device=model.device, dtype=model.dtype)
        
        print(f"[🧠] Encoding {len(texts)} texts with batch size {batch_size}...")
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                encoded = encode_with_eos(
                    model=model,
                    tokenizer=tokenizer,
                    texts=batch,
                    max_length=max_length,
                    batch_size=batch_size,
                    prefix=prefix
                )
            for idx, v in enumerate(encoded):
                norm = np.linalg.norm(v)
                if np.isnan(v).any():
                    raise ValueError(f"NaN in vector at index {i + idx}: {ids[i + idx]}")
                if norm < 1e-6:
                    raise ValueError(f"Zero vector at index {i + idx}: {ids[i + idx]}")
            vecs.append(encoded)

        vecs = np.concatenate(vecs, axis=0)

        if pca_path is not None:
            vecs = apply_pca(name, vecs, pca_path, cache_dir)

        if normalize:
            print("[📏] Normalizing vectors...")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / norms

        np.save(cache_path, vecs)
        print(f"[💾] Saved vectors to {cache_path}")

    return vecs


def build_faiss_index(
    vecs: np.ndarray,
    index_path: Optional[Union[str, Path]] = None,
    rebuild: bool = False,
    nlist: int = 1024
) -> faiss.IndexIVFFlat:
    if index_path is not None and Path(index_path).exists() and not rebuild:
        print(f"[✅] Loading FAISS index from {index_path}")
        return faiss.read_index(str(index_path))

    print("[🏗️] Building FAISS IVF index...")
    d = vecs.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    faiss.normalize_L2(vecs)
    assert index.is_trained == False
    index.train(vecs)
    index.add(vecs.astype(np.float32))

    if index_path is not None:
        faiss.write_index(index, str(index_path))
        print(f"[💾] Saved FAISS index to {index_path}")

    return index


