## full supervised
CUDA_VISIBLE_DEVICES=2,3 python main.py \
     --port 4235 \
     --world-size 2 \
     --batch-size 4 \
     --output-dir checkpoints/dhd \
     --pretrained GroundingDINO/checkpoints/groundingdino_swinb_cogcoor.pth \
     --model dhd \
     --use_cache_box \
    #  --training_type rfuc \




     
