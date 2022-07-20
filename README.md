# Decoupled Adversarial Contrastive Learning for Self-supervised Adversarial Robustness (Accepted by ECCV2022 oral presentation)

Chaoning Zhang, Kang Zhang, Chenshuang Zhang, Axi Niu, Jiu Feng, Chang D. Yoo, In So Kweon


Overall framework of DeACL. It consists of two stages. At stage 1, DeACL performs a standard SSL to obtain a non-robust encoder. At stage 2, the pretrained encoder act as a teacher model to generate pseudo-targets for guiding a supervised AT on a student model. After two stages of training, the student model is the model of our interest.
<img src="./figure/DeACL.png" width="800">

---
See also our other works:

Dual Temperature Helps Contrastive Learning Without Many Negative Samples: Towards Understanding and Simplifying MoCo (Accepted by CVPR2022) [code](https://github.com/ChaoningZhang/Dual-temperature.git) [paper](https://arxiv.org/abs/2203.17248)

---

# 🔧 Enviroment

Please refer [solo-learn](https://github.com/vturrisi/solo-learn) to install the enviroment for detail.

> First clone the repo.
> 
> Then, to install solo-learn with Dali and/or UMAP support, use:
> 
> `pip3 install .[dali,umap,h5] --extra-index-url https://developer.download.nvidia.com/compute/redist`

# ⚡ Training

## 1. prepare the pretrained self-supervised model
You can download pretrained checkpoint from [solo-learn](https://github.com/vturrisi/solo-learn) or train by yourself.

## 2. Train DeACL with ResNet18 on CIFAR10 dataset
Using the file `bash_files/pretrain/DeACL/deacl.sh`.

## 3. Test the robustness of PGD and AutoAttack with full precision 

`python adv_slf.py --ckpt CKPT_PATH`


This code is developed based on [solo-learn](https://github.com/vturrisi/solo-learn) for training and [AdvCL](https://github.com/LijieFan/AdvCL.git) for testing.

<!-- # Citation
```
@article{zhang2022dual,
  title={Dual temperature helps contrastive learning without many negative samples: Towards understanding and simplifying moco},
  author={Zhang, Chaoning and Zhang, Kang and Pham, Trung X and Niu, Axi and Qiao, Zhinan and Yoo, Chang D and Kweon, In So},
  journal={CVPR},
  year={2022}
}
``` -->

<!-- The code is coming soon. Please contact through chaoningzhang1990@gmail.com for early access. -->
