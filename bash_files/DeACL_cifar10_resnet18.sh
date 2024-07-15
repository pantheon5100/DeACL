export WANDB_API_KEY="" 

python3 main_pretrain_AdvTraining.py \
    --dataset cifar10 \
    --backbone resnet18 \
    --data_dir ./data \
    --max_epochs 100 \
    --gpus 1 \
    --accelerator gpu \
    --precision 32 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.5 \
    --classifier_lr 0.5 \
    --weight_decay 5e-4 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name "res18_simclr-cifar10-fp32" \
    --save_checkpoint \
    --method mocov2_kd_at \
    --limit_val_batches 0.2 \
    --distillation_teacher "simclr_cifar10" \
    --trades_k 2 
