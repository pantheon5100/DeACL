
# WEIGHT_DECAY="1e-6 1e-5 1e-4 2e-4 4e-4"
WEIGHT_DECAY="5e-4"

for weight_decay in $WEIGHT_DECAY; do
python3 main_pretrain_AdvTraining.py \
    --dataset cifar10 \
    --backbone resnet18 \
    --data_dir /dev/shm \
    --max_epochs 100 \
    --gpus 3 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.5 \
    --classifier_lr 0.5 \
    --weight_decay $weight_decay \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name "res18_simclr-cifar10-k_0" \
    --project DeACL \
    --entity kaistssl \
    --wandb \
    --save_checkpoint \
    --method mocov2_kd_at \
    --limit_val_batches 0.2 \
    --distillation_teacher "simclr_cifar10" \
    --trades_k 0
done




