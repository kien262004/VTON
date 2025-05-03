CUDA_VISIBLE_DEVICES=0 python -u main.py --logdir models/oc --pretrained_model pretrain/v2-1_768-ema-pruned.ckpt --base configs/viton512.yaml --scale_lr False
