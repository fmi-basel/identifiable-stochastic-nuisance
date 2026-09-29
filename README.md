# Info-LDM

This repository implements the fixed-covariance Gaussian latent predictive model developed in [].


## Installation

We recommend installing using `uv sync`.

## Setting up datasets

For Causal3DIdent and the MuJoCo hopper experiments additonal datasets have to be downloaded.

### Causal3DIdent

The Causal3DIdent experiment builds on the dataset in the [von Kügelgen et al. paper](https://arxiv.org/abs/2106.04619). We replace the exact agreement between paired views by a conditional Gaussian transition.
Download `trainset.tar.gz` and `testset.tar.gz` from the
[official Causal3DIdent release](https://zenodo.org/records/4784282), then extract
them so that the root passed to the program (standard is `./datasets/`) has this layout:

```text
Causal3DIdent/
  trainset/
    raw_latents_0.npy
    images_0/00000.png
    ...
  testset/
    raw_latents_0.npy
    images_0/00000.png
    ...
```

`wget https://zenodo.org/records/4784282/files/trainset.tar.gz`

`wget https://zenodo.org/records/4784282/files/testset.tar.gz`


### DM-Control hopper with visual nuisance

The `dm_control_hopper` dataset builds on the [DM control suite](https://github.com/google-deepmind/dm_control). 
The provided package expects to find the cloned repository at `../dm_control`.

It also expects a pretrained controller provided by the [LAOM project](https://github.com/dunnolab/laom). 
The cloned repository is expected at `../laom`.

The train and test data can be pregenerated with `./generate_mujoco_hopper_data.sh`.


## Reproducing results

All experiments can be run using the commands in the submit scripts. Evaluation notebooks can be found in `./eval/`.


## Lean

We also provide `Lean` code for the main theorem in the paper, which can be found at `./Lean/`.



