#!/usr/bin/env bash

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

vocab_size=3000
corpus_dir="/data/hwayeon/corpus"

log "Running tokenizer.sh"

# librispeech only
# lang_dir=data/libri_bpe_${vocab_size}

# if [ -f "${lang_dir}/lowered_transcript_words.txt" ]; then
#     log "File exists: ${lang_dir}/lowered_transcript_words.txt"
# else
#     log "File not found: ${lang_dir}/lowered_transcript_words.txt"
#     exit 1


# librispeech + text
lang_dir=data/libri_text_bpe_${vocab_size}
libri_dir=data/libri_bpe_500
mkdir -p $lang_dir

python3 preprocess/train_bpe_model.py \
--lang-dir $lang_dir \
--vocab-size $vocab_size \
--speech-transcript $libri_dir/lowered_transcript_words.txt \
--text-transcript $corpus_dir \
--nthread 64


log "Check status of tokenizer"
cp $libri_dir/lowered_transcript_words.txt $corpus_dir/lowered_transcript_words.txt

python3 preprocess/analyze_token.py \
--tokenizer-path $lang_dir/unigram_${vocab_size}.model
--corpus-dir $corpus_dir
--outfile "corpus_stats.json"


rm $corpus_dir/lowered_transcript_words.txt