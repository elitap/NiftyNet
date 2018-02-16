[NiftyNet][nifty] fork with code to generate the results of:

Tappeiner E, Pöll S, Hönig M, Raudaschl FP, Zaffino P, Spadea FM, Sharp CG, Schubert R, Fritscher K (2018) Efficient Multi-Organ Segmentation of the Head and Neck area using Hierarchical Neural Networks. Submitted to CARS.

#### Dataset
The data used can be downloaded [here][dataset].

#### Pre-processing
To get one image file containing the segmentation of all organs of interest, originally distributed in different files *combine_dataset_label_files.py* from the *scipts* directory can be executed. With the rescale option the CT images and the resulting segmentation are rescaled to the median spacing of the dataset (1.1mm,1.1mm,3mm).

* `python scripts/combine_dataset_label_files.py --datasetpath path-to-full-dataset-containing-the-39-datasets --outdir data/HaN_MICCAI2015_Dataset/full_dataset --rescale`

The foreground mask used by the sampler of the coarse training can be extracted using *calc_foreground_label_otsu.py*.

* `python scripts/calc_foreground_label_otsu.py --datasetpath data/HaN_MICCAI2015_Dataset/full_dataset`

For the fist stage of our hierarchical approach we train on the dataset with different spatial resolutions. The copies of the dataset are generated using the *rescale.py* script.

*  `python scripts/rescale.py --datasetpath data/HaN_MICCAI2015_Dataset/full_dataset --result data/HaN_MICCAI2015_Dataset/full_dataset_half --scale 2`
*  `python scripts/rescale.py --datasetpath data/HaN_MICCAI2015_Dataset/full_dataset --result data/HaN_MICCAI2015_Dataset/full_dataset_quarter --scale 4`
 
#### Coarse Stage Training

After the pre-processing the dataset, the different experimental configurations of the [NiftyNet][nifty] can be trained. Using the *train_configs.py* script all the configurations listed in the config file are trained. The individual configurations can be found in *configs/coarse_stage_configs*. The separation of the dataset into training and validation samples is defined in *data/HaN_MICCAI2015_Dataset/test_data_split.csv* 

* `python train_configs.py --config scripts_configs/train_config.txt --gpu 0 --stage coarse`

#### Coarse Stage Inference
 
For an intermediate evaluation of the coarse stage the segmentation results on the validation set are generated and evaluated using:

* `python infer_trained_configs.py --config scripts_configs/inference_config.txt --gpu 0 --stage coarse --checkpoint 50000 --splitfile data/HaN_MICCAI2015_Dataset/test_data_split.csv`
* `python scripts/rescale.py --datasetpath coarse_stage --checkpoint 50000`
* `python scripts/collect_results.py --modeldir coarse_stage --gtdir data/HaN_MICCAI2015_Dataset/full_dataset --resultfile coarse_res.csv --checkpoint 50000 --useplastimatch`
* `python scripts/evaluate_results.py --resultfile coarse_res.csv`

Finally, the results of the coarse stage produced by the different configurations can be found in *coarse_res_evaluated.csv* and *coarse_res_evaluated_organwise.csv*

To use the coarse network to generate the foreground mask used by the sampler of the fine stage, the pre-trained model of the best coarse configuration available in this repository *(coarse_stage/half_e-3_48-8_dice_1024s)* or any other model of the coarse stage can be used.

* `python infer_trained_configs.py --config scripts_configs/inference_config.txt --gpu 0 --stage coarse --checkpoint 100000 --splitfile data/HaN_MICCAI2015_Dataset/infer_all_for_fine_stage.csv`
* `python scripts/rescale.py --datasetpath coarse_stage --checkpoint 100000`
* `python scripts/create_maskfrom_labels.py --data coarse_stage/half_e-3_48-8_dice_1024s/output/100000 --outdir data/HaN_MICCAI2015_Dataset/mask_fine_stage` 

#### Fine Stage Training

Similar as for the coarse stage, different fine stage configuration files (available under *config/fine_stage*) can be trained by listing them inside the *train_config.txt*. The fine stage uses the result of the coarse stage as a binary mask for the sampling process.

* `python train_configs.py --config scripts_configs/train_config.txt --gpu 0 --stage fine`

#### Fine Stage Inference

The final evaluation results, with a postprocessing step after the inference can be obtained using:

* `python infer_trained_configs.py --config scripts_configs/inference_config.txt --gpu 0 --stage fine --checkpoint 50000 --splitfile data/HaN_MICCAI2015_Dataset/test_data_split.csv`
* `python scripts/postprocess.py --resdir fine_stage/selected_config/output/50000 --postprocdir fine_stage/selected_config/output/50000_post --foreground data/HaN_MICCAI2015_Dataset/mask_fine_stage`
* `python scripts/collect_results.py --modeldir fine_stage --gtdir data/HaN_MICCAI2015_Dataset/full_dataset --resultfile fine_res.csv --checkpoint 50000_post --useplastimatch`
* `python scripts/evaluate_results.py --resultfile fine_res.csv`
    
[nifty]: https://github.com/NifTK/NiftyNet
[dataset]: http://www.imagenglab.com/newsite/pddca/
