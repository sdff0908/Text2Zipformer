import os
from pathlib import Path
from tqdm import tqdm
import re

# ==== 경로 설정 ====
corpus_dir = Path("/data/hwayeon/openwebtext")  # TXT 파일들이 있는 폴더

# 전처리: HTML 태그 제거용 간단 패턴 (길이 계산만 하니까 최소화)
RE_TAG = re.compile(r"<[^>]+>")

lengths = []

def iter_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            # 태그 제거
            text = RE_TAG.sub(" ", text)
            lengths.append(len(text))

# 모든 파일 순회
for txt_file in tqdm(corpus_dir.rglob("*.txt")):
    iter_lines(txt_file)

total_lines = len(lengths)
print(f"총 라인 수: {total_lines:,}")

for max_len in [2000, 3000, 4000]:
    over_count = sum(1 for l in lengths if l > max_len)
    pct = (over_count / total_lines) * 100
    print(f"max_len={max_len}: {over_count:,} 라인 ({pct:.4f}%) 잘림")
