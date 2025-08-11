# -*- coding: utf-8 -*-
"""
Web text → ASR pretrain용 정제 파이프라인 (최종)
- HTML/스크립트/코드 제거
- URL/이메일/해시 삭제
- 유니코드/개행 표준화(단락 유지 옵션)
- 허용 문자 정책(일반/이arnings 모드)
- 과도 구두점 압축, 고립된 아포스트로피 정리
- 길이 컷/짧은 라인 드롭

사용 예:
    clean_line(raw_text, mode="earnings",
               lowercase=True, keep_paragraph_break=True)
"""

import re
import unicodedata
import ftfy

# -------- precompiled patterns --------
RE_SCRIPT      = re.compile(r"<script\b[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
RE_STYLE       = re.compile(r"<style\b[^>]*>.*?</style>",   re.DOTALL | re.IGNORECASE)
RE_TAG         = re.compile(r"<[^>]+>")  # bs4 없을 때 태그 제거 fallback
RE_CODEBLK     = re.compile(r"```.*?```", re.DOTALL)        # markdown code block
RE_INLINE_CODE = re.compile(r"`[^`]+`")                     # inline code
RE_URL         = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_EMAIL       = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_HASH        = re.compile(r"\b[a-f0-9]{32,64}\b", re.IGNORECASE)  # 긴 해시/uuid류
RE_MULTI_PUNCT = re.compile(r"([!?.,'-])\1{2,}")           # 과도한 반복 구두점
# 추가 패턴
RE_PRE_BLOCK  = re.compile(r"<pre\b[^>]*>.*?</pre>", re.DOTALL | re.IGNORECASE)
RE_CODE_BLOCK = re.compile(r"<code\b[^>]*>.*?</code>", re.DOTALL | re.IGNORECASE)
RE_TILDE_FENCE= re.compile(r"~~~.*?~~~", re.DOTALL)  # ``` 말고 ~~~ 펜스
RE_SHEBANG    = re.compile(r"^#!", re.MULTILINE)
RE_CODE_KW    = re.compile(r"\b(import|from|def|class|#include|public|static|package|namespace|using\s+namespace|return|fn|function|var|let|const|template)\b")
RE_ARROW_OP   = re.compile(r"->|=>|::|:=")

# 도메인별 허용 문자셋
ALLOWED_GENERAL = re.compile(r"[^a-zA-Z0-9 .,?!'\n]")
ALLOWED_EARN    = re.compile(r"[^a-zA-Z0-9 .,?!'/$%:+\-():\n]")  # Earnings-22용

def strip_html(text: str) -> str:
    """스크립트/스타일/태그 제거 (bs4 없이)"""
    text = RE_SCRIPT.sub(" ", text)
    text = RE_STYLE.sub(" ", text)
    text = RE_TAG.sub(" ", text)
    return text


def normalize_unicode(
    text: str,
    *,
    lowercase: bool = False,
    keep_paragraph_break: bool = True,
) -> str:
    """유니코드/개행/스마트 기호 정규화"""
    if not text:
        return ""

    # 0) 유니코드 복원/정규화
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)

    # 1) 개행 표준화
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if keep_paragraph_break:
        # 3개 이상 개행 → 2개로 축소
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 단일 개행은 공백으로 (양쪽이 \n이 아닌 \n만 치환)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        # 문단 경계 주변 공백 정리
        text = re.sub(r"\s*\n\n\s*", "\n\n", text)
    else:
        # 모든 개행을 공백으로
        text = text.replace("\n", " ")

    # 2) 특수문자 정리
    text = text.replace("\uFFFD", "")  # replacement char

    # 3) 제어문자 제거 (개행은 위에서 처리했으므로 제외)
    ctrl_map = {c: None for c in range(0x00, 0x20) if c not in (0x0A,)}  # 0x0A = \n
    text = text.translate(ctrl_map)

    # 4) 스마트쿼트/대시/말줄임표 정규화
    text = (text.replace("“", "\"").replace("”", "\"")
                 .replace("‘", "'").replace("’", "'")
                 .replace("—", "-").replace("–", "-")
                 .replace("…", "..."))

    if lowercase:
        text = text.lower()

    return text


def is_code_like_line(line: str) -> bool:
    # 기호 비율
    sym = re.findall(r"[;{}()\[\]<>:=+\-*/%&|^~]", line)
    sym_ratio = len(sym) / max(len(line), 1)
    # JSON/YAML 의심
    jsonish = (line.strip().startswith(("{","[")) and line.count(":") >= 1)
    # 코드 키워드/화살표/세미콜론 다수 중 2개 이상 걸리면 코드 취급
    signals = 0
    signals += RE_CODE_KW.search(line) is not None
    signals += RE_ARROW_OP.search(line) is not None
    signals += (line.strip().endswith(("{","}",";")))
    signals += (sym_ratio > 0.30)  # 필요시 0.25~0.35 튜닝
    signals += jsonish
    signals += (RE_SHEBANG.search(line) is not None)

    return signals >= 2


def clean_line(
    text: str,
    *,
    mode: str = "general",          # "general" | "earnings"
    drop_ratio: float = 0.3,        # 허용 외 문자 비율 임계
    lowercase: bool = True,         # ASR용이면 True 권장
    keep_paragraph_break: bool = True,
    max_len: int = 4000,            # 너무 긴 라인 컷 (이상치 방지)
    min_len: int = 5,               # 너무 짧은 라인 드롭
) -> str:
    if not text or text.isspace():
        return ""

    # 0) 광고/코드/마크업 우선 제거
    text = RE_CODEBLK.sub(" ", text)           # ```...```
    text = RE_TILDE_FENCE.sub(" ", text)       # ~~~...~~~
    text = RE_INLINE_CODE.sub(" ", text)       # `inline`
    text = RE_PRE_BLOCK.sub(" ", text)         # <pre>...</pre> 내용 통째 제거
    text = RE_CODE_BLOCK.sub(" ", text)        # <code>...</code> 내용 통째 제거
    text = strip_html(text)                    # 남은 태그 제거

    # 1) URL/이메일/해시 삭제
    text = RE_URL.sub("", text)
    text = RE_EMAIL.sub("", text)
    text = RE_HASH.sub("", text)

    # 라인 단위 코드스멜 드롭 (문서/문단을 한 번에 넣는다면 split 후 필터)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    kept = []
    for ln in lines:
        if is_code_like_line(ln):
            continue
        kept.append(ln)
    text = "\n".join(kept)

    # 2) 유니코드/개행/소문자 정규화
    text = normalize_unicode(text, lowercase=lowercase, keep_paragraph_break=keep_paragraph_break)

    # 2.5) 너무 긴 라인 컷 (옵션)
    if max_len and len(text) > max_len:
        text = text[:max_len]

    # 3) 허용 문자 정책 적용 + 드롭 판정
    allowed = ALLOWED_EARN if mode == "earnings" else ALLOWED_GENERAL
    if text:
        bad = len(allowed.findall(text))
        if bad / max(len(text), 1) > drop_ratio:
            return ""

    # 허용 외 문자는 공백으로 치환
    text = allowed.sub(" ", text)

    # 3.5) 고립된 아포스트로피 정리 (don't, i'm은 유지 / 낱개 ' 제거)
    text = re.sub(r"(?<!\w)\'(?!\w)", " ", text)

    # 4) 과도한 구두점 반복 압축 (최대 3개)
    text = RE_MULTI_PUNCT.sub(r"\1\1\1", text)

    # 5) 공백 정리 + 길이 필터
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < min_len:
        return ""

    return text




# import os
# from pathlib import Path
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed

# SRC = Path("/data/hwayeon/openwebtext")
# DST = Path("/data/hwayeon/filtered_openwebtext")
# N_WORKERS = os.cpu_count() * 2  # I/O 병렬: 코어수의 2배 정도

# def process_one(src_path: Path):
#     try:
#         with src_path.open("r", encoding="utf-8", errors="ignore") as f:
#             text = f.read()
#         text = clean_line(text, mode="earnings")
#         if not text:
#             return False

#         rel = src_path.relative_to(SRC)
#         dst_path = DST / rel
#         dst_path.parent.mkdir(parents=True, exist_ok=True)
#         with dst_path.open("w", encoding="utf-8") as f:
#             f.write(text)
#         return True
#     except Exception as e:
#         # 필요시 로깅
#         print(f"ERROR {src_path}: {e}")
#         return False

# def iter_files(root: Path):
#     for p in root.rglob("*"):
#         if p.is_file():
#             yield p


# def chunkify(files: list, chunk_size: int = 5000):
#     for i in range(0, len(files), chunk_size):      
#         yield files[i:i+chunk_size]    
        

# def main():
#     filtered = 0
#     files = list(iter_files(SRC))
#     with ThreadPoolExecutor(max_workers=N_WORKERS) as ex, \
#         tqdm(total=len(files), desc="Processing files", unit="file") as pbar:
#         for chunk in chunkify(files, 5000):
#             futures = [ex.submit(process_one, p) for p in chunk]
#             for fut in as_completed(futures):
#                 flag = fut.result()  # True/False 등 반환하면 필요시 집계
#                 if not flag:
#                     filtered += 1   
#                 pbar.update(1)

#     print(f"{filtered} files are filtered among {len(files)} files: This is {100*filtered/total:.2f} %")
# if __name__ == "__main__":
#     main()