#!/usr/bin/env bash

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

vocab_sizes=3000
corpus_dir=/path/to/corpus

log "Running tokenizer.sh"

# librispeech only
lang_dir=data/libri_bpe_${vocab_size}

if [ -f "${lang_dir}/lowered_transcript_words.txt" ]; then
    log "File exists: ${lang_dir}/lowered_transcript_words.txt"
else
    log "File not found: ${lang_dir}/lowered_transcript_words.txt"
    exit 1

python3 preprocess/train_bpe_model.py \
--lang-dir $lang_dir \
--vocab-size $vocab_size \
--speech-transcript $lang_dir/lowered_transcript_words.txt 


# text only 
lang_dir=data/text_bpe_${vocab_size}
mkdir -p $lang_dir

python3 preprocess/train_bpe_model.py \
--lang-dir $lang_dir \
--vocab-size $vocab_size \
--text-transcript $corpus_dir


# librispeech + text
lang_dir=data/libri_text_bpe_${vocab_size}
mkdir -p $lang_dir

python3 preprocess/train_bpe_model.py \
--lang-dir $lang_dir \
--vocab-size $vocab_size \
--speech-transcript $lang_dir/lowered_transcript_words.txt \
--text-transcript $corpus_dir
