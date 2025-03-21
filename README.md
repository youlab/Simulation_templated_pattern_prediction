# Physics_constrained_DL_pattern_prediction
![Under Construction](https://img.shields.io/badge/Status-Under%20Construction-orange)

For the manuscript:  Physics constrained photorealistic prediction of self-organized P. aeruginosa patterns 

The code is divided into three folders
1) ***seed_to_sim1_sim2_deterministic***
2) ***sim_to_exp_diffusion***
3) ***sim_generation***

In the current version of the paper these three folder correspond to the following tasks-
1) **Pre-trained Stable Diffusion VAE for image compression and training NN on the compressed latent embeddings**

We use this pipeline to demonstrate orders of magnitude acceleration using emulated NN models. We use dilated ResNets inspired from [PDEarena](https://github.com/pdearena/pdearena?tab=readme-ov-file), Stable Diffusion v1-4 Autoencoder from [Huggingface Diffusers Library](https://huggingface.co/CompVis/stable-diffusion-v1-4). This pipeline is used to demonstrate prediction of simulation end-points from initial seeds and also to test the data demand for the second part by testing mapping between two simulated conditions. 

2) **ControlNet attached to Stable Diffusion for spatial conditioning the U-Net on images of patterns**

We use this pipeline to demonstrate generative pattern prediction of experiments that have been conditioned on simulations. Major part of the code for ControlNet is borrowed from [here](https://github.com/lllyasviel/ControlNet), with additional modifications being made for preprocessing images and inference pipeline. This pipeline is used to demonstrate probablisitic prediction of experiment end-point patterns by conditioning on simulated end-point patterns. Current version uses the pre-trained v1-5 Stable Diffusion.  

3) **MATLAB codes for generation of simulation end-point pattern pairs**

Contains the details of generating simulations with varying end-point patterns with parallel processing using CPUs for generating a large number of patterns. Code is modified from Nan Luo's MSB 2021 [Paper](https://www.embopress.org/doi/full/10.15252/msb.202010089) [Code](https://github.com/youlab/OptimalPatterns_NanLuo) 

### Details of code files inside each folder:
1) ***seed_to_sim1_sim2_deterministic***
- *latent_generation folder*: Scripts for generating latent from end-point patterns, done in a seperate script and saved as pickle files to be used during model training. 
- *slurm_scripts*: Files for running python jobs on Duke Computing Cluster(DCC)
- `SDVAE_Simulation.ipynb` and `Display_seed_sim_exp.ipynb` are the scripts for displaying the out of box accuracy of Stable Diffusion v1-4 VAE in compressing biological simulated patterns of varying intial seeding configurations and displaying a sample of expreimental patterning under different seeding configurations and comparision with simulations with seed, simulation and experimental images.  **Fig 1**
- `Training_seedtointermediate_dilRESNETs.ipynb` and `Prediction_seedtointermediate_dilResNet.ipynb` are the training and inference notebooks for prediction of simulations(intermediate denotes simulations of a certain parameter configuration) from intial seeding configurations. **Fig 2 and Supplementary Fig 8**
- `Training_intermediatetocomplex_dilResNets.ipynb` and `Prediction_intermediatetocomplex_dilResNet.ipynb` are training and inference notebooks for mapping of simulations from one parameter configuration(intermediate in the naming convention used) to another(complex). **Fig 3 and Supplementary Fig 12**
- `Training_datademand_intermediatetocomplex.py` and `Prediction_datademand_intermediatetocomplex.ipynb` are training and inference files for assessing the data demand requirement when mapping between two parameters, a proxy for mapping between simulations and experiments in the next part. **Fig 4**
- `Training_simtoexp_dilRESNET.ipynb` and `Prediction_SimtoExp_dilResNet.ipynb` are training and inference files for predicting experimental patterns from simulated patterns, demonstration of the need for probabilistic models for predicting stochastic experimental pattern formation. **Supp Fig 16 and Supplementary Fig 17**

2) ***sim_to_exp_diffusion***

- *annotator*, *ldm*, *models* subfolders are unchanged from the original ControlNet code
- *cldm* contains two new scripts logger_custom.py and preprocess.py for adding custom saving of images from trained models and preprocessing simulation or experimental images respectively
- *Slurm_scripts* subfolder contains the slurm scripts for execution of python files in DCC
- `config.py` and `share.py` are unchanged from original implementation
- `tool_add_control.py` , `tool_add_control_sd21` and `tool_transfer_control.py` are also unchanged, and are required for adding ControlNet to Stable Diffusion- details of how to do this are explained later. 
- `simtoexp_dataset.py` creates the dataset for mapping from simulation to processed color images of experimental patterns
- `simtoexp_train.py` Training using the above dataset using the v1-5 Stable Diffusion + ControlNet 
- `inference_simtoexp.py` Inference using the above trained model
- Similar as the orientation of the above 3 files, `simtoexp_BnW_dataset.py`, `simtoexp_BnW_train.py` and `inference_simtoexp_BnW.py` are files for creating dataset, training and inference for predicting the black and white processed image of experimental colony patterns. 
- `Display_finalresults_inference.ipynb` contains the display script for displaying the orignal seed, predicted image using the trained ControlNet and the ground truth experiments for Simulation to Experiment Black and White and Color Prediction. **Fig 5 and Fig 6**
- `inference_gradio.ipynb` is the gradio web interface for mapping from simulations to Black and White experimental patterns (currently the input needs a processed cropped version of the simulated input)
- `image_to_seed.py`, `vae_util.py`,`predict_util.py` ,`models.py` and `prediction_seedtosim.py` are the scripts for getting an image-> convert to a seeding configuration-> use the saved dilResNet model and predict the simulated patterns. We want to integrate this with the gradio pipeline so a web interface would allow the user to run both the parts of my pipeline- 1) seed to simulation and 2) simulation to experiment. 
- `inference_gradio_start_to_end.ipynb` is the notebook for running the gradio from start(seed) to end (predicted experimental patterns)- currently still under construction, facing OOM issues on DCC. 

3) ***sim_generation***
(Still under construction- original files, not well commented)
- *Slurm_scripts* subfolder contains the slurm scripts for execution of MATLAB files in DCC(current version still involved using CPUs)
- `Optimal_Patterns_3Tp2Cond.m` is the MATLAB script that generates the simulations-- currently it generates 2 sets of patterns for the 2 sets of simulation configurations we have in the paper. It also generates the patterns for 3 Timepoints, and as to why we do it, it's out of scope of the current paper but related with modelling temporal dynamics too in the next version of the paper. 
- `Optimal_Patterns_3Tp2Cond_ModelTesting.m` is the MATLAB script for generating simulation in the testing dataset. This testing dataset has some pre-defined patterning grids that are defined in the `designs_24_10_02.mat` (generated from `Img_design.m`). The purpose of this is to test the performance of the model on some grids it has not seen before, like pre-defined designs which might have denser configurations than the random seeding configurations in the training set which have a large possible grid for the seed distribution. 
- `Patterns_generator_experimental.m` is the script used to generate 1000 random seeding grids for experimental MANTIS seeding configurations. Corresponding to this we generate analogus simulation seeds in <couldn't find the file, generate again from an earlier version of code> .These simulated input seeding grids are saved in `simulatedPatterns.mat`, and are simulated using the DCC in `Optimal_Patterns_ExperimentalCondns.m`. Note that the way in which the input seeding grids for experiments were generated, they resulted in a random configuration. Similar procedure is applied for fixed, pre-defined seeding configurations that are stored in `simulatedPatterns_Fixed.mat` and simulation seeds are generated in `ExptoSim_MAT.m` with end-point patterns stored in `Optimal_Patterns_ExperimentalCondns_Fixedgrid.m`. 
- `Parameters_multiseeding.mat` and `Branching_diffusion.m` are files unchanged from the original implementation in the MSB 2021 paper, essentially containing various parameters for estabilishing certain parameters and diffusion processes in the model. 

### Note: 
1) Currently the code still has dependency on large files which are cumbersome to attach to the repository. However, the people who can access this current repository can also access the files, mostly which are stored in the /hpc/group/youlab/ks723 folders. I would enable access through large file hosting databases in the future. The current version of all the code files listed in this repo are stored in the /hpc/dctrl/ks723 folders though and you might not be able to access it. But this repo contains all the essential code files and other instructions to install large essential files are outlined in the next step. 
2) To run codes from the 2)sim to exp diffusion folder, you have to essentially install two large files. Firstly, you have to download the [Stable Diffusion v1-5 checkpoint](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt) and place in the same folder. Secondly for attaching the ControlNet to the pre-trained network, run 
    ```
    python tool_add_control.py v1-5-pruned.ckpt control_sd15_ini.ckpt
    ```
    Steps can be modified in the similar manner by running `tool_add_control_sd21.py` for attaching Stable diffusion to the v2-1 ckpt. These both were supported in the original ControlNet implementation. 
3) All of the python libraries that were used to run scripts from both folders are in the  conda environment file pytorch_PA_patternprediction.yml file(might contain unnecessary libraries to the current code implementation). Run
    ```
    conda env create -f pytorch_PA_patternprediction.yml
    ```
    















