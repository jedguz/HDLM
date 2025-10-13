<div align="center">

<h1>Next Semantic Scale Prediction via 
    Hierarchical Diffusion Language Models | NeurIPS 2025</h1>

<div>
    <a href="https://homepage.zhouc.ai/" target="_blank">Cai Zhou</a><sup>1,*</sup> | 
    <a href="https://chenyuwang-monica.github.io/" target="_blank">Chenyu Wang</a><sup>1,*</sup> | 
    <a href="https://zdhnarsil.github.io/" target="_blank">Dinghuai Zhang</a><sup>2,*</sup> | 
    <a href="https://shangyuantong.github.io/" target="_blank">Shangyuan Tong</a><sup>1</sup> | 
    <a href="https://yifeiwang77.com/" target="_blank">Yifei Wang</a><sup>1</sup> |
    <a href="https://stephenbates19.github.io/index.html" target="_blank">Stephen Bates</a><sup>1,†</sup> |
    <a href="https://people.csail.mit.edu/tommi/tommi.html" target="_blank">Tommi Jaakkola</a><sup>1,†</sup>
</div>
<br>
<div>
    <sup></sup><sup>1</sup> Massachusetts Institute of Technology<br><sup>2</sup> Microsoft Research
</div>
<br>
<div>
    <sup>*</sup> Equal Contribution<br><sup>$\dagger$</sup> Equal Senior Supervision
<br>


[![arXiv](https://img.shields.io/badge/arXiv-2510.08632-b31b1b.svg)](https://arxiv.org/abs/2510.08632)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-4b44ce.svg)](https://neurips.cc/virtual/2025/poster/116380)


<div align="left"> 

## 📢 News
- [2025/09/18] HDLM is accepted to NeurIPS 2025!
- [2025/10/12] Paper is available on arXiv!
- [2025/10/12] Code is released!

## 💻 Overview
<br>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="HDLM.png">
</div>
<br>

We present Hierarchical Diffusion Language Model (GIDD), a novel framework for training discrete diffusion models via time-varying next-semantic scale prediction.
HDLM extends standard Masked Diffusion Model (MDM) by introducing intermediate hierarchies (termed cluster tokens) in between clean tokens and masked tokens.
In the forward process, each token is independently perturbed to its higher-level ancestor with more abstract semantics according to the scheduler, while in the reverse process the model progressively predicts the next, more detailed semantics. 
Taken together, HDLM provides a general time-varying next semantic scale prediction process for language modeling. We derive closed-form expressions for the diffusion Evidence Lower Bound (ELBO), and show that HDLM can be implemented in a flexible manner while including the existing MDM as a special case.
This repository contains all training and evaluation code necessary for reproducing the results in the paper.


## 🔧 Quick Start

Set up the environment:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

## 🎈 Reproducing Experiments

### Training

#### Precomputing cluster dicts and embeddings
You can download our precalculated files in `hdlm/clusters` for existing numbers of clusters in [1, 2, 4, 8, 16, 32, 64, 128, 256] with [GPT-2 tokenizer](https://huggingface.co/openai-community/gpt2) on [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset using [GIDD](https://arxiv.org/abs/2503.04482) pretrained models, or preprocess by running `hdlm/compute_cluster.py` for customed numbers of clusters / tokenizers / datasets / pretrained models. Make sure the names / paths of these cluster files match `cluster_dict_path`, `cluster_embed_path` and `pretrained_model_name` in your training configs as in the examples.

#### Configs

To reproduce the training runs from the paper, you can use the following commands.
In this example, we are training on a single node with 8 GPUs, feel free to adjust the `--nnodes` and `--nproc_per_node` arguments to match your setup.

Whenever needed, feel free to change the checkpoint saving directory by adjusting `save_dir` in `hdlm/configs/logging/default.yaml`, and data storage directory by `cache_dir` in `hdlm/configs/data/defaults.yaml`.

Key hyperparameters include:
 * `cluster_size`: number of clusters ($n$ in the paper)
 * `gamma`: forward process schedule ($\gamma$ in the paper)
 * `p_perturb`: probability of stochastic perturbations ($1-\xi$ in the paper)

 You are also welcome to try out other model / training / loss hyperparameters.

(optional) Log into W&B with `wandb login` for experiment tracking or other disable via `wandb disabled`.

```bash
# HDLM-small-64
torchrun --nnodes 1 --nproc_per_node 8 hdlm/train.py --config-name hdlm-small-cluster_64-gamma_1.0-xi_1.0 logging.run_name="'small-hdlm-cluster_64-gamma_1.0-xi_1.0-owt'"

# GIDD+ baseline
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name gidd logging.run_name="'small-gidd+-owt-pu=0.0'"

# MDLM baseline
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name mdlm logging.run_name="'small-mdlm-owt'"

# AR baseline
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name ar logging.run_name="'small-ar-owt'"
```


### Evaluation

There are also a couple of scripts to run inference and evaluate the trained models.

#### Generate samples
The following command will generate `num_samples=256` samples in `num_denoising_steps=512` iterations from the model checkpoint located at `path` and save them to `samples_dir=samples.pt`.
```bash
python hdlm/eval/generate_samples.py path=./outputs/path/to/checkpoint/ samples_dir=samples.pt num_samples=256 num_denoising_steps=512 batch_size=16
```

#### Generative PPL
Given a file containing samples generated with the `generate_samples.py` script, the following command will compute the generative PPL.
Here we assume that the diffusion model used to generate samples located at `samples.pt` uses the `gpt2` tokenizer, and we compute generative PPL using `gpt2-large` as a reference model.
The results will be saved to `metrics_path=metrics.json`.
```bash
python hdlm/eval/generative_ppl.py samples_path=samples.pt model_tokenizer=gpt2 pretrained_model=gpt2-large batch_size=1 metrics_path=metrics.json
```

#### Validation loss
A simple helper script to compute the loss of a trained model on the entire validation split.
```bash
python hdlm/eval/loss.py path=./outputs/path/to/checkpoint/ batch_size=32
```


## 📎 Citation 

If you find our work helpful, please consider giving a star ⭐ and citation 📝 
```bibtex
@article{zhou2025next,
  title={Next Semantic Scale Prediction via Hierarchical Diffusion Language Models},
  author={Zhou, Cai and Wang, Chenyu and Zhang, Dinghuai and Tong, Shangyuan and Wang, Yifei and Bates, Stephen and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2510.08632},
  year={2025}
}
```

## 💞 Acknowledgements
The code is built upon the below repositories, we thank all the contributors for open-sourcing.
* [GIDD](https://github.com/dvruette/gidd/)
* [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
