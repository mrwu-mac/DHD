CUDA_VISIBLE_DEVICES=1 python inference.py \
    --resume checkpoints/stage2-hico-vcoco-vg/ckpt_115616_16.pt \
    --stage 2 \
    --index 3 \
    --action 1 \
    --image-path /assets/test1.jpeg