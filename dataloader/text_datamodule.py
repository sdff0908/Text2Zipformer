from typing import Iterable, List, Dict, Iterator
import os

import sentencepiece as spm

import torch
from torch.utils.data import IterableDataset, get_worker_info

from icefall.utils import AttributeDict

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
    ):
        super().__init__()
        self.file_list = self._collect_text(params.corpus_path)
        if not self.file_list:
            raise FileNotFoundError(f"No files matched: {paths}")
        self.sp = sp
        self.params = params

        self.bos_id = self.sp.piece_to_id("<bos>")
        self.eos_id = self.sp.piece_to_id("<sos>")
    # ---------- helpers ----------
    @staticmethod
    def _collect_text(corpus_dir: str) -> List[str]:
        out = []
        for root, dirs, files in os.walk(corpus_dir):
            for f in files:
                out.append(os.path.join(root, f))

        return out

    @staticmethod
    def _open(fp: str) -> Iterator[str]:
        f = open(fp, "rt", encoding="utf-8", errors="ignore")
        try:
            for line in f:
                yield line
        finally:
            f.close()

    @staticmethod
    def _ddp_info():
        # 안전하게 DDP 정보 획득
        world_size = 1
        rank = 0
        if dist is not None and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            rank = int(os.environ.get("RANK", "0"))
        return rank, world_size

    def _assign_files(self) -> List[str]:
        # 파일을 (DDP 랭크 × 워커) 전체 샤드에 modulo 분배
        worker = get_worker_info()
        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        rank, world = self._ddp_info()
        global_worker_id = rank * num_workers + worker_id
        total_shards = world * num_workers
        return [f for i, f in enumerate(self.file_list) if (i % total_shards) == global_worker_id]

    # ---------- core ----------
    def __iter__(self) -> Iterable[torch.Tensor]:
        files = self._assign_files()
        # 토큰 버퍼: concat 후 seq_len+1 고정 길이 조각 산출
        buf: List[int] = []

        def _flush_chunks():
            # 버퍼에서 seq_len+1 토큰씩 잘라 yield
            nonlocal buf
            while len(buf) >= (self.params.seq_len + 1):
                chunk = buf[: self.params.seq_len + 1]
                buf = buf[self.params.seq_len : ]  # 다음 조각을 위해 seq_len만큼 소비 (겹치기 1)
                yield torch.tensor(chunk, dtype=torch.long)

        for fp in files:
            for raw in self._open(fp):
                assert isinstance(raw, str)
                ids = self.sp.encode(raw)
                # BOS/EOS 부착 (문장 단위)
                if self.params.add_bos:
                    buf.append(self.bos_id)  # type: ignore[arg-type]
                buf.extend(ids)
                if self.params.add_eos:
                    buf.append(self.eos_id)  # type: ignore[arg-type]

                # 가능한 한 많이 방출
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
    def __init__(self, pad_token_id: int, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 각 시퀀스 길이
        lengths = [len(x) for x in batch]
        # teacher-forcing 위해 한 토큰씩 줄여서 정렬 (input:[:-1], label:[1:])
        eff_lengths = [max(0, L - 1) for L in lengths]
        max_len = max(eff_lengths) if eff_lengths else 0

        B = len(batch)
        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, max_len), self.ignore_index, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)

        for i, ids in enumerate(batch):
            L = len(ids)
            if L <= 1:
                continue
            src = ids[:-1]      # 입력(마지막 제외)
            tgt = ids[1:]       # 정답(첫 토큰 제외)
            T = len(src)
            input_ids[i, :T] = src
            labels[i, :T] = tgt
            attention_mask[i, :T] = 1

        return {
            "input_ids": input_ids,           # (B, T)
            "labels": labels,                 # (B, T)  (CrossEntropyLoss에 바로 사용)
            "attention_mask": attention_mask, # (B, T)
            "lengths": torch.tensor(eff_lengths, dtype=torch.long),
        }