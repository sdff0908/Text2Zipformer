from typing import Iterable, List, Dict, Iterator
import os
import logging

import sentencepiece as spm

import torch
from torch.utils.data import IterableDataset, get_worker_info

from icefall.utils import AttributeDict, setup_logger

try:
    import torch.distributed as dist
except Exception:
    dist = None

class TextIterableDataset(IterableDataset):
    """
    대용량 텍스트를 라인 단위로 스트리밍해 토크나이즈하고,
    토큰을 이어붙여(seq concat) 길이 seq_len+1 조각으로 잘라서 산출.
    (collate에서 [-1]/[1:] 시프트해 NTP 학습)

    - 파일 포맷: .txt(라인별 문장)
    - gzip(.gz)/xz(.xz) 자동 인식
    - 멀티워커/분산(DDP) 샤딩: 파일 단위 modulo 분배
    """

    def __init__(
        self,
        params: AttributeDict,
        sp: spm.SentencePieceProcessor,
        file_list: List,
        rank: int,
        world_size: int,
    ):
        super().__init__()
        self.file_list = file_list
        self.sp = sp
        self.params = params

        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

        self.rank = rank
        self.world_size = world_size

    # ---------- helpers ----------
    @staticmethod
    def _open(fp: str) -> Iterator[str]:
        f = open(fp, "rt", encoding="utf-8", errors="ignore")
        try:
            for line in f:
                yield line
        finally:
            f.close()

    def _shard(self, paths: List[str]):
        # world_size: splitted number
        total = len(paths)
        per_rank = total // self.world_size
        remainder = total % self.world_size
        start = self.rank * per_rank + min(self.rank, remainder)
        end = start + per_rank + (1 if self.rank < remainder else 0)
        sharded_paths = paths[start:end]
        return sharded_paths, (start, end)

    def _assign_files(self) -> List[str]:
        sharded_files, (start_idx, end_idx) = self._shard(self.file_list)
        return sharded_files, start_idx, end_idx

    # ---------- core ----------
    def __iter__(self) -> Iterable[torch.Tensor]:
        files, start_idx, end_idx = self._assign_files()

        setup_logger(f"{self.params.exp_dir}/log/log-data")
        logging.info(f"[RANK {self.rank}: Assigned {end_idx-start_idx-1} files from {start_idx} to {end_idx-1} among {len(self.file_list)} files")

        # 토큰 버퍼: concat 후 seq_len+1 고정 길이 조각 산출
        buf: List[int] = []

        def _flush_chunks():
            # 버퍼에서 seq_len+1 토큰씩 잘라 yield
            nonlocal buf
            while len(buf) >= (self.params.seq_len + 1):
                chunk = buf[: self.params.seq_len + 1]
                buf = buf[self.params.seq_len: ]  # 다음 조각을 위해 seq_len만큼 소비 (겹치기 1)
                yield torch.tensor(chunk, dtype=torch.long)

        for fp in files:
            for raw in self._open(fp):
                assert isinstance(raw, str)
                ids = self.sp.encode(raw)
                # BOS/EOS 부착 (문서 단위)
                buf.append(self.bos_id)
                buf.extend(ids)
                buf.append(self.eos_id)

                for chunk in _flush_chunks():
                    yield chunk

        # 파일 끝: 남은 버퍼 처리
        if not self.params.drop_last and len(buf) > 1:
            # 길이를 seq_len+1로 패딩해도 되고, 그냥 남은 길이 그대로 내보내도 됨
            yield torch.tensor(buf, dtype=torch.long)
            

class LMCollator:
    """
    LM용 collate:
    - input_ids:  [B, T-1]  (원문에서 마지막 토큰 제외)
    - labels:     [B, T-1]  (원문에서 첫 토큰 제외)
    - attention_mask: [B, T-1]
    패딩은 tokenizer.pad_token_id 사용, labels의 패딩은 -100(ignore_index)
    """
    def __init__(self, seq_len: int, pad_id: int, bos_id: int, ignore_index: int = -100):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.ignore_index = ignore_index
        
    def __call__(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        input_ids = torch.full((B, self.seq_len), self.pad_id, dtype=torch.long)
        labels = torch.full((B, self.seq_len), self.ignore_index, dtype=torch.long)

        for i, ids in enumerate(batch):
            T = len(ids)
            if T <= 1:
                continue
            if T == self.seq_len + 1: 
                input_ids[i] = ids[:-1]  # 입력(마지막 제외)
                labels[i] = ids[1:]      # 정답(첫 토큰 제외)
            else:
                t = T - 1
                input_ids[i, :t] = ids[:-1]
                labels[i, :t] = ids[1:]

        labels.masked_fill_(input_ids == self.bos_id, self.ignore_index)  # BOS는 예측 제외
        
        return {
            "input_ids": input_ids,           # (B, T)
            "labels": labels,                 # (B, T)  (CrossEntropyLoss에 바로 사용)
        }
        
    
def count_iterable_dl(dl):
    cnt = 0
    for _ in dl:
        cnt += 1
    return cnt