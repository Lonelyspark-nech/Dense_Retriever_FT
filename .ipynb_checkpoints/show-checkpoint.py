from pathlib import Path

def show_head(path, num_lines=5):
    path = Path(path)
    print(f"\n📄 文件: {path.name}")
    if not path.exists():
        print("❌ 文件不存在")
        return
    with open(path, 'r', encoding='utf-8') as f:
        for i in range(num_lines):
            line = f.readline()
            if not line:
                break
            print(line.strip())

# 修改文件名以切换数据集
show_head("./datasets/dev_small_queries_input.jsonl")
show_head("./datasets/dev_small_qrels.tsv")
show_head("./datasets/corpus.tsv")
show_head("./datasets/train_triplets.jsonl")
