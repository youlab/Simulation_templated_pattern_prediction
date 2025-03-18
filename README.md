# Physics_constrained_DL_pattern_prediction
For the manuscript:  Physics constrained photorealistic prediction of self-organized P. aeruginosa patterns 

The code is divided into two folders
- seed_to_sim1_sim2_deterministic
- sim_to_exp_diffusion
- sim_generation

In the current version of the paper these three folder correspond to the following tasks-
1) **Pre-trained Stable Diffusion VAE for image compression and training NN on the compressed latent embeddings**
We use this pipeline to demonstrate orders of magnitude acceleration using emulated NN models. We use dilated ResNets inspired from [PDEarena](https://github.com/pdearena/pdearena?tab=readme-ov-file), Stable Diffusion v1-4 Autoencoder from [Huggingface Diffusers Library](https://huggingface.co/CompVis/stable-diffusion-v1-4). This pipeline is used to demonstrate prediction of simulation end-points from initial seeds and also to test the data demand for the second part by testing mapping between two simulated conditions. 

2) **ControlNet attached to Stable Diffusion for spatial conditioning the U-Net on images of patterns**
We use this pipeline to demonstrate generative pattern prediction of experiments that have been conditioned on simulations. Major part of the code for ControlNet is borrowed from [here](https://github.com/lllyasviel/ControlNet), with additional modifications being made for preprocessing images and inference pipeline. This pipeline is used to demonstrate probablisitic prediction of experiment end-point patterns by conditioning on simulated end-point patterns. Current version uses the pre-trained v1-5 Stable Diffusion.  

3) **MATLAB codes for generation of simulation end-point pattern pairs**
Contains the details of generating simulations with varying end-point patterns with parallel processing using CPUs for generating a large number of patterns. Code is modified from Nan Luo's MSB 2021 [Paper](https://www.embopress.org/doi/full/10.15252/msb.202010089) [Code](https://github.com/youlab/OptimalPatterns_NanLuo) 

### Details of code files inside each folder:
1) seed_to_sim1_sim2_deterministic
- *latent_generation folder*: Scripts for generating latent from end-point patterns, done in a seperate script and saved as pickle files to be used during model training. 
- *slurm_scripts*: Files for running python jobs on Duke Computing Cluster
- `SDVAE_Simulation.ipynb` and `Display_seed_sim_exp.ipynb` are the scripts for displaying the out of box accuracy of Stable Diffusion v1-4 VAE in compressing biological simulated patterns of varying intial seeding configurations and displaying a sample of expreimental patterning under different seeding configurations and comparision with simulations with seed, simulation and experimental images.  **Fig 1**
- `Training_seedtointermediate_dilRESNETs.ipynb` and `Prediction_seedtointermediate_dilResNet.ipynb` are the training and inference notebooks for prediction of simulations(intermediate denotes simulations of a certain parameter configuration) from intial seeding configurations. **Fig 2 and Supplementary Fig 8**
- `Training_intermediatetocomplex_dilResNets.ipynb` and `Prediction_intermediatetocomplex_dilResNet.ipynb` are training and inference notebooks for mapping of simulations from one parameter configuration(intermediate in the naming convention used) to another(complex). **Fig 3 and Supplementary Fig 12**
- `Training_datademand_intermediatetocomplex.py` and `Prediction_datademand_intermediatetocomplex.ipynb` are training and inference files for assessing the data demand requirement when mapping between two parameters, a proxy for mapping between simulations and experiments in the next part. **Fig 4**
- `Training_simtoexp_dilRESNET.ipynb` and `Prediction_SimtoExp_dilResNet.ipynb` are training and inference files for predicting experimental patterns from simulated patterns, demonstration of the need for probabilistic models for predicting stochastic experimental pattern formation.**Supp Fig 16 and Supplementary Fig 17**

 
   






