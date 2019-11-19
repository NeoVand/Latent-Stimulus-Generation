# Latent-Stimulus-Generation
This notebook allows generation of multi-modal stimuli from BigGanDeep 512 and StyleGAN

## Installation
1. Download and Install Anaconda from: https://www.anaconda.com/distribution/
2. Clone the repository in a directory:
```console
git clone https://github.com/NeoVand/Latent-Stimulus-Generation.git
```
3. Create a python 3.7 environment by typing this command in the Anaconda Prompt (or any shell with conda in PATH):
```console
cd Flower2
conda env create --name stim --file environment.yml
```
4. Activate the environment:
```console
conda activate mneflower
```
5. Download the pretrained weights for StyleGAN from [here](https://github.com/lernapparat/lernapparat/releases/download/v2019-02-01/karras2019stylegan-ffhq-1024x1024.for_g_all.pt) and copy it in the folder

## Running
1. run `Jupyter notebook` in the folder (use an Administrator terminal if on windows) and then open the `stimgen` notebook.
