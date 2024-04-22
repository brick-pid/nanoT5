export HF_ENDPOINT=https://hf-mirror.com

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 python -m nanoT5.main \
    optim.grad_acc=12 \
    optim.batch_size=48 \
    optim.base_lr=2e-3 \
    data.corpus=cj_java_mix \
    model.name=Salesforce/codet5p-770m \
    data.mix_ratio=6 \

#     optim.name={adafactor,adamwscale} \
#     optim.lr_scheduler={legacy,cosine}

# NCCL_P2P_DISABLE=1 accelerate launch -m nanoT5.main \
#     optim.batch_size=16