export HF_ENDPOINT=https://hf-mirror.com

python -m nanoT5.main \
    optim.name={adafactor,adamwscale} \
    optim.lr_scheduler={legacy,cosine}