from collections import Counter
from pathlib import Path
import sentencepiece as spm
import re, json
from tqdm import tqdm

def iter_lines(root):
    root = Path(root)
    for p in root.rglob("*"):
        if p.is_file():
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

def token_stats_spm(spm_model_path, data_dir, max_lines=None):
    sp = spm.SentencePieceProcessor(model_file=spm_model_path)
    cnt = Counter()
    total_tokens = 0
    total_words = 0
    unk_id = sp.unk_id()

    unk_cnt = 0
    for i, line in enumerate(tqdm(iter_lines(data_dir), desc="Tokenizing")):
        if max_lines and i >= max_lines:
            break
        ids = sp.encode(line, out_type=int)
        total_tokens += len(ids)
        total_words += len(re.findall(r"\S+", line))
        cnt.update(ids)
        unk_cnt += sum(1 for x in ids if x == unk_id)

    # 상위 100
    top = []
    for i, c in cnt.most_common(100):
        top.append({
            "token": sp.id_to_piece(i),
            "count": c,
            "token_rate": c / total_tokens
        })

    stats = {
        "total_tokens": total_tokens,
        "unk_rate": (unk_cnt / total_tokens) if total_tokens else 0.0,
        "top100": top
    }

    return stats

if __name__ == "__main__":
    s = token_stats_spm("/path/to/asr_unigram_2k.model", "/data/corpus/openwebtext_filtered")
    with open("spm_token_stats.json", "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
