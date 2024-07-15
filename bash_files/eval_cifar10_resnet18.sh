# For SLF 
# python adv_finetune.py \
#     --ckpt DEACL_WEIGHT_res18_simclr-cifar10-offline-x4h7cp45-ep=99.ckpt \
#     --mode slf

# For AFF
# python adv_finetune.py \
#     --ckpt DEACL_WEIGHT_res18_simclr-cifar10-offline-x4h7cp45-ep=99.ckpt \
#     --mode aff

# For ALF
# python adv_finetune.py \
#     --ckpt DEACL_WEIGHT_res18_simclr-cifar10-offline-x4h7cp45-ep=99.ckpt \
#     --mode alf


CKPT="trained_models/simclr/DEACL_WEIGHT_res18_simclr-cifar10-offline-x4h7cp45-ep=99.ckpt"
# run the SLF 5 times
for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=1 python adv_finetune.py \
        --ckpt $CKPT \
        --mode slf \
        --learning_rate 0.1
done
