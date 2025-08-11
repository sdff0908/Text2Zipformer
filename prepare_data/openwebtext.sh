#!/usr/bin/env bash

repo_dir=/workspace/zipformer_text/openwebtext #/path/to/tar_dir
corpus_dir=/data/hwayeon/corpus #/path/to/corpus_dir

python3 preprocess/download_text_corpus.py \
    --repo-dir $repo_dir \
    --corpus-dir $corpus_dir