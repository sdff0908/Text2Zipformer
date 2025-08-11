# TITLE


## 0. Set environment
Based on docker file: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
```bash
# k2
pip install k2==1.24.4.dev20240725+cuda12.1.torch2.4.0 -f https://k2-fsa.github.io/k2/cuda.html

# lhotse
pip install git+https://github.com/lhotse-speech/lhotse

# Set sys.path
export PYTHONPATH="/path/to/zipformer_text:$PYTHONPATH"
```



## 1. Prepare data
This process includes downloading data and computing feature bank for further audio-text pair training

```python
# Text
apt-get install git-lfs
git lfs install --skip-smudge 
git clone git@hf.co:datasets/Skylion007/openwebtext

cd openwebtext/
git remote set-url origin https://<user_name>:<token>@huggingface.co/datasets/Skylion007/openwebtext
git lfs pull

cd ../prepare_data
bash openwebtext.sh

# LibriSpeech
bash librispeech.sh

# Tedlim3
bash tedlium3.sh

# Earnings22
bash earnings22.sh
```

## 2. Train CTC layer with text data



## 3. Train model with audio-text data



## 4. Evaluate