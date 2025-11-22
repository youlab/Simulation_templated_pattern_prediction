# Physics_constrained_DL_pattern_prediction
![Maintenance](https://img.shields.io/badge/maintenance-active-brightgreen)

For the manuscript:  Physics constrained photorealistic prediction of self-organized P. aeruginosa patterns by Kinshuk Sahu, Harris M. Davis, Jia Lu, César A. Villalobos, Avi Heyman, Emrah Şimşek and Lingchong You

See Notes section at the end for information about execution of these various scripts. 

The code is divided into three folders
1) ***seed_to_sim1_sim2_deterministic***
2) ***sim_to_exp_diffusion***
3) ***sim_generation***

In the current version of the paper these three folder correspond to the following tasks-
1) **Pre-trained Stable Diffusion VAE for image compression and training NN on the compressed latent embeddings**

We use this pipeline to demonstrate orders of magnitude acceleration using emulated NN models. We use dilated ResNets inspired from [PDEarena](https://github.com/pdearena/pdearena?tab=readme-ov-file), Stable Diffusion v1-5 Autoencoder from [Huggingface Diffusers Library](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). This pipeline is used to demonstrate prediction of simulation end-points from initial seeds and also to test the data demand for the second part by testing mapping between two simulated conditions. 

2) **ControlNet attached to Stable Diffusion for spatial conditioning the U-Net on images of patterns**

We use this pipeline to demonstrate generative pattern prediction of experiments that have been conditioned on simulations. Major part of the code for ControlNet is borrowed from [here](https://github.com/lllyasviel/ControlNet), with additional modifications being made for preprocessing images and inference pipeline. This pipeline is used to demonstrate probablisitic prediction of experiment end-point patterns by conditioning on simulated end-point patterns. Current version uses the pre-trained v1-5 Stable Diffusion.  

3) **MATLAB codes for generation of simulation end-point pattern pairs**

Contains the details of generating simulations with varying end-point patterns with parallel processing using CPUs for generating a large number of patterns. Code is modified from Nan Luo's MSB 2021 [Paper](https://www.embopress.org/doi/full/10.15252/msb.202010089) [Code](https://github.com/youlab/OptimalPatterns_NanLuo) 

### Details of code files inside each folder:
1) ***seed_to_sim1_sim2_deterministic***
- *latent_generation folder*: Scripts for generating latent from end-point patterns, done in a seperate script and saved as pickle files to be used during model training. 
- *slurm_scripts*: Files for running python jobs on Duke Computing Cluster(DCC)
- *models*: Contains implemention of the two models used in this first pipeline, `vae.py` and `dilResNet.py`
- *utils*: Comprises of `config.py`, `display.py` and `preprocess.py`. `config.py` describes the physical locations of various files generated and imported in this pipeline.  `display.py` contains various display functions used for different figures in the manuscript. `preprocess.py` has various functions that preprocess the raw files before being used in the rest of the pipeline. 
- `Display_seed_sim_exp_Fig1.ipynb` is the scripts for displaying the out of box accuracy of Stable Diffusion v1-5 VAE in compressing biological simulated patterns of varying intial seeding configurations and displaying a sample of experimental patterning under different seeding configurations and comparision with simulations with seed, simulation and experimental images.  **Fig 1**
- `SSIM_ExpandSim_SuppFig1.ipynb` compares the reconstruction image quality using the SD VAE on the simulated and experimental data. **Supp Fig 1**
- `Image_grids_SuppFig2_3_4_8_9_13_14.ipynb` has the script for displaying various sample 10x10 image grids starting from input seed, simulated patterns (default), latent of simulated patterns, simulated patterns (Condition 2), latent of simulated patterns (Condition 2), experimental patterns and  corresponding simulated patterns.  **Supp Fig 2,3,4,8,9,13,14** 
- `Training_seedtointermediate_dilRESNETs.py` , `Prediction_seedtointermediate_dilResNet_CPUbenchmark_SuppInf.ipynb` `Prediction_seedtointermediate_dilResNet_Fig2_SuppFig6.ipynb` are the training and inference notebooks for prediction of simulations(intermediate denotes simulations of a certain parameter configuration) from intial seeding configurations. **Fig 2 and Supp Fig 6**
- `Scatterplot_SSIM_seedvariation_SuppFig7.ipynb` describes the SSIM variation of the predictions of the dilResNet model on the test set (compared with ground truth simulations) as a function of initial seeding number. **Supp Fig 7**
- `Training_intermediatetocomplex_dilResNets.py` and `Prediction_intermediatetocomplex_dilResNet_Fig3_SuppFig10.ipynb` are training and inference notebooks for mapping of simulations from one parameter configuration(intermediate in the naming convention used) to another(complex). **Fig 3 and Supp Fig 10**
- `Image_preprocessing_Expimages.ipynb` is the script describing data preprocessing and data augmentation techniques for simulation and experimental images.  Actual data augmentation codes are `Augmentation_ExpandSim.py` and `Augmentation_ExpandSim_optimized.py`. The optimized version generates 40k patterns in about 6 hours, the non-optimized version was used in the paper and takes around 2.5 days. Augmented patterns from both of the versions are similar and is verified in the `Image_preprocessing_Expimages.ipynb` notebook. You would have to run these codes to generate the augmented datasets that are used in the sim_to_exp_diffusion pipeline.   **Supp Fig 11**
- `Training_datademand_intermediatetocomplex.py`and   `Prediction_datademand_intermediatetocomplex_Fig4.ipynb` , are training and inference files for assessing the data demand requirement when mapping between two parameters, a proxy for mapping between simulations and experiments in the next part. `Training_datademand_inttocomplex_augmentation.py` and `Prediction_datademand_augmentation_intermediatetocomplex_Fig4.ipynb` are training and inference files for similar data demand assessment, but now with data augmentation to 40k patterns. `DataDemand_Augmentation.py` contains the augmentation code for different unique simmulated patterns  **Fig 4**
- `Training_simtoexp_dilResNets.py` and `Prediction_simtoexp_dilResNet_SuppFig12.ipynb` are training and inference files for predicting experimental patterns from simulated patterns, demonstration of the need for probabilistic models for predicting stochastic experimental pattern formation. **Supp Fig 12**

2) ***sim_to_exp_diffusion***

- *annotator*, *ldm*, *models* subfolders are unchanged from the original ControlNet code
- *cldm* contains two new scripts logger_custom.py and preprocess.py for adding custom saving of images from trained models and preprocessing simulation or experimental images respectively
- *Slurm_scripts* subfolder contains the slurm scripts for execution of python files in DCC
- `config.py` and `share.py` are unchanged from original implementation
- `tool_add_control.py` , `tool_add_control_sd21` and `tool_transfer_control.py` are also unchanged, and are required for adding ControlNet to Stable Diffusion- details of how to do this are explained later. 
- `create_promptjson.ipynb` is the script for creating a json file that has the information about source and target image pairs and the blank text prompt, used in training of ControlNet. 
- `simtoexp_dataset.py` creates the dataset for mapping from simulation to processed color images of experimental patterns
- `simtoexp_train.py` Training using the above dataset using the v1-5 Stable Diffusion + ControlNet
- `batch_infer.py` Inference using the above trained model, `pipeline.py` describes how the trained checkpoint is loaded and describes the inference pipeline. 
- `plot_Fig5.ipynb` contains the display script for displaying the orignal seed, predicted image using the trained ControlNet and the ground truth experiments for Simulation to Experiment mapping. **Fig 5**
- Similar as the orientation of the `simtopexp_dataset.py`, `simtoexp_train.py` , `batch_infer.py` and `pipeline.py` files, `seedtoexp_dataset.py`, `seedtoexp_train.py`, `batch_infer_seedtoexp.py` and `pipeline_seedtoexp.py` are the files for creating dataset, training and inference for predicting the experimental images starting from the initial seeds instead of the simulations as a baseline estimate. To create the augmentation dataset which has the input seed configurations rotated corresponding to the experiments we have the file `Seed_DataAugmentation.py`. **Supp Fig 15**
- `batch_infer_ablation.py` and `batch_infer_seedsweep.py` are the scripts that show how the inference depends on some controllable parameters, like negative prompt, positive prompt etc and also the intial starting random seeding configuration.  **Supp Fig 16**
- `DissimilarityScore_Seeding` is the script where the training of the Siamese network takes place, for calculating the dissimilarity between sets of predictions, simulations and experimental images, and to probe how the inter replicate variation compares to the one between different seeding patterns. **Supp Fig 17 and Supp Fig 18**
- `inference_gradio.ipynb` is the gradio web interface for mapping from simulations to experimental patterns.   
- `inference_quantmetrics_seedtoexp.ipynb` and `inference_quantmetrics_simtoexp.ipynb` are scripts for computing the various image comparision metrics between simulations, experiments and diffusion model predictions. **Supp Table 1**


3) ***sim_generation***
(Still under construction- original files, not well commented)
- *Slurm_scripts* subfolder contains the slurm scripts for execution of MATLAB files in DCC(current version still involved using CPUs)
- `Optimal_Patterns_3Tp2Cond_vmod.m` is the MATLAB script that generates the simulations-- currently it generates 2 sets of patterns for the 2 sets of simulation configurations we have in the paper. It also generates the patterns for 3 Timepoints, and as to why we do it, it's out of scope of the current paper (related with modelling temporal dynamics) . 
- `Optimal_Patterns_3Tp2Cond_ModelTesting.m` is the MATLAB script for generating simulation in the testing dataset. This testing dataset has some pre-defined patterning grids that are defined in the `designs_24_10_02.mat` (generated from `Img_design.m`). The purpose of this is to test the performance of the model on some grids it has not seen before, like pre-defined designs which might have denser configurations than the random seeding configurations in the training set which have a large possible grid for the seed distribution. 
- `Patterns_generator_experimental.m` is the script used to generate 1000 random seeding grids for experimental MANTIS seeding configurations. Corresponding to this we generate analogus simulation seeds in `SimulatedGridMATgeneration.m` .These simulated input seeding grids are saved in `simulatedPatterns.mat`, and are simulated using the DCC in `Optimal_Patterns_ExperimentalCondns.m`. Note that the way in which the input seeding grids for experiments were generated, they resulted in a random configuration. Similar procedure is applied for fixed, pre-defined seeding configurations that are stored in `simulatedPatterns_Fixed.mat` and simulation seeds are generated in `ExptoSim_MAT.m` with end-point patterns stored in `Optimal_Patterns_ExperimentalCondns_Fixedgrid.m`. 
- `Parameters_multiseeding.mat` and `Branching_diffusion.m` are files unchanged from the original implementation in the MSB 2021 paper, essentially containing various parameters for estabilishing certain parameters and diffusion processes in the model. 

### Note: 
1) Currently the code still has dependency on large files like image datasets, latent pickle objects, saved models etc which are cumbersome to attach to the repository. 

 [Updated 06 Nov 2025]: All the files that are needed to run the seed to simulation part(Simulation, experiment and seed images,also saved trained model files) are available in the public Huggingface datasets [here](https://huggingface.co/datasets/HotshotGoku/Physics_constrained_DL_pattern_prediction).

    Steps to follow (for generating figures from paper/ running only the inference pipeline) :

    i) Download files from Huggingface datasets. Extract the tar files. 

    ii) Change the locations in the config files to whereever you stored these files. Note that there are 2 config files, for seed_to.... the file is located under utils folder and in the sim_to_exp... folder, the file is located under the cldm folder. 

    iii) Run the latent generation codes `latent_from_complex.py` and `Latent_from_final_patterns_intermediate.py` to save the pickle files. After these are done, you can go and update the locations in the config file under the utils folder. 

    iv) Run the prediction codes with the saved model files. To see basic prediction performance used in Figure 2,3 and 5 run the scripts: `Prediction_seedtointermediate_dilResNet_Fig2_SuppFig6.ipynb`, `Prediction_intermediatetocomplex_dilResNet_Fig3_SuppFig10.ipynb`, `batch_infer.py` and `plot_Fig5.ipynb` For figure 4, you can run `Prediction_datademand_intermediatetocomplex_Fig4.ipynb ` and `Prediction_datademand_augmentation_intermediatetocomplex_Fig4.ipynb`. 

    v) The slurm scripts that were used to run the python files on the Duke computing cluster have been attached for reference. You can modify the gpu nodes and file locations accordingly. 

    vi) (Optional) For training the models in the deterministic pipeline,you can work with what you have for Fig 2 and 3. For Fig 4, you would have to run the DataDemand_Augmentation.py code first, edit wherever your files are saved in the config file. Then run the `latent_complex_dataaugmentation.py` and `latent_intermediate_dataaugmentation.py`. Some of these files have a taskID in the code to run parallel jobs, if you do not have SLURM, replace these accordingly in the code. 
    For training the models in the sim_to_exp_diffusion pipeline (Fig 5), it is imperative to run the Experimental+ Simulation augmentation script (`Augmentation_ExpandSim_optimized.py`) to generate ~40k patterns for experimental-simulation dataset from the base ~400 patterns. Change the locations in the config file. 
    If you want to generate the Supplementary Figure 7 in the paper, you would have to run the Experimental+Simulation augmentation script from the point above first, then run `latent_from_Exp_images.py` and `latent_from_SimcorrtoExp_images.py`. Change the locations in the config file.
     

2) To run codes from the 2)sim to exp diffusion folder, you have to essentially install two large files. Firstly, you have to download the [Stable Diffusion v1-5 checkpoint](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt) 

    ```
    from huggingface_hub import hf_hub_download

    # Repository and file details
    repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"  
    filename = "v1-5-pruned.ckpt"  

    # Specify the directory where the file should be downloaded
    local_dir = "./Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/"                # Change to your desired directory

    # Download the file
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

    print(f"File downloaded to: {file_path}")
    ```
and place in the same folder. Secondly for attaching the ControlNet to the pre-trained network, run 
```
python tool_add_control.py v1-5-pruned.ckpt ./control_sd15_ini.ckpt
```
Steps can be modified in the similar manner by running `tool_add_control_sd21.py` for attaching Stable diffusion to the v2-1 ckpt. These both were supported in the original ControlNet implementation. 

3) All of the python libraries that were used to run scripts from both folders are in the  conda environment file pytorch_PA_patternprediction.yml file(might contain unnecessary libraries to the current code implementation). Run
    ```
    conda env create -f pytorch_PA_patternprediction.yml
    ```
    



