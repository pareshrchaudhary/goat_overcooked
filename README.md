# Improving Human-AI Coordination through Online Adversarial Training and Generative Models

## Description

Training framework of **GOAT**: *G*enerative **O**nline **A**dversarial **T**raining

paper: [Improving Human-AI Coordination through Online Adversarial Training and Generative Models](https://arxiv.org/abs/2504.15457)

demo/website: [https://sites.google.com/view/goat-2025/home](https://sites.google.com/view/goat-2025/home)



## Get Started

## Install

First you need to have a python environment with pytorch. If you don't have, here is an example of creating a vitual conda environment:

```bash
conda create -n goat python=3.12
conda activate goat
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

And then install other packages. These (and a few missing) packages suffice to work on top of an minimal pytorch virtual environment (conda/docker)

```bash
pip install absl-py dill numpy scipy tqdm gym pettingzoo ipython pygame ipywidgets opencv-python
pip install wandb icecream setproctitle seaborn tensorboardX slackweb psutil slackweb pyastar2d einops h5py
```

To install module "mapbt" for MARL and "overcooked" for Overcooked!

```bash
git clone https://github.com/pareshrchaudhary/goat_overcooked.git
# Initialize and update submodules (required for overcooked environment and population data)
git submodule update --init --recursive
# Move to the root directory
cd mapbt
# Install mapbt
pip install -e .
# Install overcooked
pip install -e mapbt/envs/overcooked/overcooked_berkeley
```



## Tips

- Never work on the "main" branch directly! Create a new branch and do whatever you want
- Don't forget "git rebase main" to keep updated with the "main" branch, especially before the pull request
- For most cases, create new files or write additional code for a new feature, rather than modify the original code



## Run (Example: Overcooked)


For a complete pipeline to train a Cooperator for overcooked, please refer to [this document](mapbt/scripts/overcooked_population/README.md)

Here we only show some simple examples.


### Self-play on Overcooked

```bash
# cd mapbt/scripts
bash train_overcooked_sp.sh 
```



### Zero-shot Coordination on Overcooked

#### Generate Config

(Optional) Use the following script to generate config files if the config files are absent:

```bash
# cd mapbt/scripts
bash generate_overcooked_policy_config.sh
```

The config files can also be found under the "run_dir" of current experiment



#### Stage 1: Population-based Training

```bash
# cd mapbt/scripts
bash train_overcooked_mep_stage1.sh
```



### Stage 2: Train the Adaptive Policy

Copy the model to this folder: "mapbt/scripts/policy_models". Specifiy the path to the policy models that will be used in stage 2 in the ``--population_yaml_path'' file

Then run

```bash
# cd mapbt/scripts
bash train_overcooked_mep_stage2.sh
```

### Evaluatiuon with Human Proxy Model

```bash
# Train the human proxy model by behavior cloning
bash train_overcooked_bc.sh
# Move the model to mapbt/scripts/policy_models
cp "mep_adaptive_model" "mapbt/scripts/policy_models/[layout_name]/mep_adaptive.pt"
cp "bc_model" "mapbt/scripts/policy_models/[layout_name]/proxy.pt"
# Evaluate
bash eval_overcooked_mep_vs_proxy.sh
```

### Train and Evaluate GOAT with Human Proxy Model

Ensure that policy models directory path is updated in the script and chose between different models for the adversary.

```bash
bash train_overcooked_regret.sh gpu_id "seed list"
```



## Cite

```
@article{chaudhary2025improving,
  title={Improving Human-AI Coordination through Adversarial Training and Generative Models},
  author={Chaudhary, Paresh and Liang, Yancheng and Chen, Daphne and Du, Simon S and Jaques, Natasha},
  journal={arXiv preprint arXiv:2504.15457},
  year={2025}
}
```

## Reference

This repository is built upon [the official implementation](https://github.com/lych1233/GAMMA-human-ai-collaboration) of [this paper](https://openreview.net/forum?id=v4dXL3LsGX)
