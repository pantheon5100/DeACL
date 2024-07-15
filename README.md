# Decoupled Adversarial Contrastive Learning for Self-supervised Adversarial Robustness

Chaoning Zhang*, Kang Zhang*, Chenshuang Zhang, Axi Niu, Jiu Feng, Chang D. Yoo, In So Kweon
(*Equal contribution)

This is the official implementation of the paper "Decoupled Adversarial Contrastive Learning for Self-supervised Adversarial Robustness," which was accepted for an oral presentation at ECCV 2022.

The DeACL framework consists of two stages. In the first stage, DeACL performs standard self-supervised learning (SSL) to obtain a non-robust encoder. In the second stage, the pretrained encoder acts as a teacher model, generating pseudo-targets to guide supervised adversarial training (AT) on a student model. The student model, trained through these two stages, is the final model of interest. DeACL is a general framework that can be applied to any SSL method and AT method.
<img src="./figure/DeACL.png" width="800">

Paper link: [arXiv](https://arxiv.org/abs/2207.10899), [ECCV2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900716.pdf), [supplementary material](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900716-supp.pdf)


# Change log
*2024.7.14* We have rewritten the SLF and AFF code to enhance its readability and usability. The new code is more modular, making it easier to extend to other datasets and models. Using this updated code, we conducted experiments with SimCLR and ResNet18 on the CIFAR10 dataset (ckpt [here](https://drive.google.com/file/d/1yc38miWGY57sHS6W6aY_k5t69Gt5v5fm/view?usp=sharing)). The code was executed five times, and the average results are reported below. The updated code can be found in the `adv_finetune.py` file.

*2023.3.2* The different definitions of the Resnet model between pre-train and SLF make the forward and backward different. Our previous code can get a different result given in the paper. We fixed the bug by changing the Resnet used during the SLF setting and released the pre-trained model with new code, which performs slightly differently from the one reported in the paper (SLF with CIFAR10 (AA,RA,SA) reported in paper: `45.31, 53.95, 80.17` -> with current code: `45.57, 55.43, 79.53`). (We apologize for not providing the model used in the paper since we accidentally deleted the original file.) We also update the environment configuration to help you reproduce our result.

# 🔧 Enviroment
We use [conda](https://docs.conda.io/en/latest/miniconda.html) for python enviroment management. After installing conda,

1. conda create -n deacl python=3.8

2. conda activate deacl

3. pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

2. pip install -r requirements.txt


# ⚡ Training and Evaluation

## 1. prepare the pretrained teacher self-supervised model
You can download pretrained checkpoint from [solo-learn](https://github.com/vturrisi/solo-learn#cifar-10) or train by yourself. 

SimCLR model pretrained by solo-learn [link](https://drive.google.com/drive/folders/1mcvWr8P2WNJZ7TVpdLHA_Q91q4VK3y8O?usp=sharing).

put the downloaded model into folder 'TeacherCKPT'

## 2. Train DeACL with ResNet18 on CIFAR10 dataset

Using the file `bash_files\DeACL_cifar10_resnet18.sh`.

You need to specific the `--project xxx`, put your wandb api key, and add `--wandb` to enable wandb logging.

## 3. Test the robustness of PGD and AutoAttack under standard linear fine-tuning (SLF) and adversarial full fine-tuning (AFF)
First install [autoattack](https://github.com/fra31/auto-attack) package `pip install git+https://github.com/fra31/auto-attack`

### a. Eval the the trained model in step 2
Use the following commandline, replace the `CKPT_PATH` with the path of the trained model.

```bash
# SLF
python adv_finetune.py --ckpt CKPT_PATH --mode SLF --learning_rate 0.1
# AFF
python adv_finetune.py --ckpt CKPT_PATH --mode AFF --learning_rate 0.01
```

### b. Eval the pretrained model provided by us
We privide our pretrained model on CIFAR10 with teacher model SimCLR at [here](https://drive.google.com/file/d/1yc38miWGY57sHS6W6aY_k5t69Gt5v5fm/view?usp=sharing), you can download it and use the commandline in a. to evaluate the model.

The results get by the pretrained model with the above code are as follows (average of 5 runs). For SLF, the initial learning rate is 0.1, and for AFF and ALF, the initial learning rate is 0.01 with beta 6 in trades loss. All three modes training epochs are 25, and the learning rate decay at 15 and 20 epochs by 10 times.
| Mode | AA | RA | SA | 
| --- | --- | --- | --- |
| SLF | 46.14 ± 0.054 | 53.45 ± 0.095 | 80.82 ± 0.090 |
| AFF | 50.75 ± 0.150 | 54.23 ± 0.238 | 83.64 ± 0.139 |
| ALF | 45.55 ± 0.134 | 55.30 ± 0.142 | 79.39 ± 0.140 |


# Acknowledgement
This code is developed based on [solo-learn](https://github.com/vturrisi/solo-learn) for training and [AdvCL](https://github.com/LijieFan/AdvCL.git), [AutoAttack](https://github.com/fra31/auto-attack) and [TRADES](https://github.com/yaodongyu/TRADES) for testing.

<!-- # Citation
```
@article{zhang2022dual,
  title={Dual temperature helps contrastive learning without many negative samples: Towards understanding and simplifying moco},
  author={Zhang, Chaoning and Zhang, Kang and Pham, Trung X and Niu, Axi and Qiao, Zhinan and Yoo, Chang D and Kweon, In So},
  journal={CVPR},
  year={2022}
}
``` -->

# See also our other works

Dual Temperature Helps Contrastive Learning Without Many Negative Samples: Towards Understanding and Simplifying MoCo (Accepted by CVPR2022) [GitHub](https://github.com/ChaoningZhang/Dual-temperature.git), [arXiv](https://arxiv.org/abs/2203.17248)



# Citation
```
@inproceedings{zhang2022decoupled,
  title={Decoupled Adversarial Contrastive Learning for Self-supervised Adversarial Robustness},
  author={Zhang, Chaoning and Zhang, Kang and Zhang, Chenshuang and Niu, Axi and Feng, Jiu and Yoo, Chang D and Kweon, In So},
  booktitle={ECCV 2022},
  pages={725--742},
  year={2022},
  organization={Springer}
}
```
