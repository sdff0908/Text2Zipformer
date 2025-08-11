import os
import re
import argparse

import ftfy
from tqdm import tqdm
import datasets
from datasets.download.download_manager import DownloadManager

from normalize_text import clean_line


class Openwebtext(datasets.GeneratorBasedBuilder):
    def __init__(self, data_dir):
        num_files = 21
        data_files = [os.path.join(data_dir, "subsets/urlsf_subset{:02d}.tar".format(i)) for i in range(num_files)]
        self.data_files = data_files
        
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            homepage="https://skylion007.github.io/OpenWebTextCorpus/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        archives = dl_manager.download(self.data_files)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                "archive_iterators": [
                    dl_manager.iter_archive(archive) for archive in archives
                ],
                "iter_archive": dl_manager.iter_archive
            }),
        ]

    def _generate_examples(self, archive_iterators, iter_archive):
        """Yields examples."""
        for archive_iterator in tqdm(archive_iterators, desc="Reading data files"):
            for xz_filepath, xz_f in archive_iterator:
                if not xz_filepath.endswith(".xz"):
                    continue
                for txt_filepath, txt_f in iter_archive(xz_f):
                    if not txt_filepath.endswith(".txt"):
                        continue
                    idx = f"{xz_filepath}/{txt_filepath}"
                    yield idx, {"text": txt_f.read().decode("utf-8").strip()}


def download_and_extract_corpus(repo_dir, corpus_dir, mode="earnings"):
    module = Openwebtext(repo_dir)
    dl_manager = DownloadManager()
    data_gen = module._split_generators(dl_manager)
    archive_iterators = data_gen[0].gen_kwargs["archive_iterators"]
    iter_archive = data_gen[0].gen_kwargs["iter_archive"]

    os.makedirs(corpus_dir, exist_ok=True)
    for idx, info_dict in module._generate_examples(archive_iterators, iter_archive):
        raw_text = info_dict["text"]

        # text normalize
        text = clean_line(raw_text, mode=mode, lowercase=True, keep_paragraph_break=True)
        if not text:
            continue

        # e.g) idx: openwebtext/urlsf_subset00-1000_data.xz/0999049-cf978a7f8ecbf43416e69377c444de07.txt
        sub_dir = os.path.join(corpus_dir, idx.split(".")[0]) 
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        with open(os.path.join(sub_dir, idx.split("/")[-1]), "w") as f:
            f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", 
                        help="Path to openwebtext")
    parser.add_argument("--corpus-dir", 
                        help="Save path for *.txt after extracting *.tar.gz")
    parser.add_argument("--mode", default="earnings", choices=["earnings", "general"],
                        help="How to normalize text")
    args = parser.parse_args()  

    download_and_extract_corpus(args.repo_dir, args.corpus_dir, args.mode)