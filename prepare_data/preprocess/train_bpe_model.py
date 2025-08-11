#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# You can install sentencepiece via:
#
#  pip install sentencepiece
#
# Due to an issue reported in
# https://github.com/google/sentencepiece/pull/642#issuecomment-857972030
#
# Please install a version >=0.1.96

import argparse
import shutil
from pathlib import Path
from typing import Dict
import logging

import sentencepiece as spm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        The generated bpe.model is saved to this directory.
        """,
    )

    parser.add_argument(
        "--text-transcript",
        type=str,
        default=None,
        help="Training transcript. Directory path to corpus",
    )

    parser.add_argument(
        "--speech-transcript",
        type=str,
        default=None,
        help="Training transcript. *.txt path to transcript"
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Vocabulary size for BPE training",
    )

    parser.add_argument(
        "--nproc",
        type=int,
        help="Number of process for training",
    )
    return parser.parse_args()


def generate_tokens(lang_dir: Path):
    """
    Generate the tokens.txt from a bpe model.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(lang_dir / "bpe.model"))
    token2id: Dict[str, int] = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}
    with open(lang_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for sym, i in token2id.items():
            f.write(f"{sym} {i}\n")


def line_iter(paths):
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def mix_iter(text_paths, speech_paths, p_libri=0.3):
    iters = [line_iter(text_paths), speech_paths]
    probs = [1-p_libri, p_libri]
    while True:
        src = random.choices([0,1], weights=probs, k=1)[0]
        try:
            yield next(iters[src])
        except StopIteration:
            break  # 실전에선 고갈된 쪽을 교체/루프 등 보완


def main():
    args = get_args()
    vocab_size = args.vocab_size
    lang_dir = Path(args.lang_dir)

    model_type = "unigram"

    model_prefix = f"{lang_dir}/{model_type}_{vocab_size}"

    logging.info(f"Transcript path: {args.speech_transcript} and {args.text_transcript}")
    if args.speech_transcript and not args.text_transcript:
        train_text = args.speech_transcript
        large_corpus = False
    elif not args.speech_transcript and args.text_transcript:
        train_text = [str(p) for p in Path(args.text_transcript).rglob("*.txt")]
        large_corpus = True
    elif args.speech_transcript and args.text_transcript:
        train_text = mix_iter(args.text_transcript, args.speech_transcript)
        large_corpus = True

    
    character_coverage = 1.0
    input_sentence_size = 100000000

    user_defined_symbols = ["<blk>", "<sos/eos>"]
    unk_id = len(user_defined_symbols)
    # Note: unk_id is fixed to 2.
    # If you change it, you should also change other
    # places that are using it.

    model_file = Path(model_prefix + ".model")
    if not model_file.is_file():
        spm.SentencePieceTrainer.train(
            input=train_text,
            vocab_size=vocab_size,
            model_type=model_type,
            model_prefix=model_prefix,
            input_sentence_size=input_sentence_size,
            character_coverage=character_coverage,
            user_defined_symbols=user_defined_symbols,
            train_extremely_large_corpus=large_corpus,
            shuffle_input_sentence=True, 
            byte_fallback=True,
            # hard_vocab_limit=False,
            num_threads=args.nproc,
            unk_id=unk_id,
            bos_id=-1,
            eos_id=-1,
        )
    else:
        print(f"{model_file} exists - skipping")
        return

    shutil.copyfile(model_file, f"{lang_dir}/bpe.model")

    generate_tokens(lang_dir)


if __name__ == "__main__":
    main()