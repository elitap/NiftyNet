[NiftyNet][nifty] fork with code to generate the results of:

Tappeiner E, Pöll S, Hönig M, Raudaschl FP, Zaffino P, Spadea FM, Sharp CG, Schubert R, Fritscher K (2018) Efficient Multi-Organ Segmentation of the Head and Neck area using Hierarchical Neural Networks. Submitted to CARS.

##### Dataset
The data used can be downloaded [here][dataset].

##### Pre-processing
To get one image file containing the segmentation of all organs of interest, originally distributed in different files *combine_dataset_label_files.py* from the *scipts* directory can be executed. With the rescale option the CT images and the resulting segmentation are rescaled to the median spacing of the dataset (1.1mm,1.1mm,3mm).

* `python script/combine_dataset_label_files.py --datasetpath path-to-full-dataset-containing-the-39-datasets --outdir data/HaN_MICCAI2015_Dataset/full_dataset --rescale`

The foreground mask used by the sampler of the coarse training can be extracted using *calc_foreground_label_otsu.py*.

* `python script/calc_foreground_label_otsu.py --datasetpath data/HaN_MICCAI2015_Dataset/full_dataset`

For the fist stage of our hierarchical approach we train on the dataset with different spatial resolutions. The copies of the dataset are generated using the *rescale.py* script.

*  `python script/rescale.py --datasetpath data/HaN_MICCAI2015_Dataset/full_dataset --result data/HaN_MICCAI2015_Dataset/full_dataset_half --scale 2`
*  `python script/rescale.py --datasetpath data/HaN_MICCAI2015_Dataset/full_dataset --result data/HaN_MICCAI2015_Dataset/full_dataset_quarter --scale 4`
 
##### Coarse Stage Training

After the pre-processing the dataset, the different experimental configurations of the [NiftyNet][nifty] can be trained. Using the *train_configs.py* script all the configurations listed in the config file are trained. The individual configurations can be found in *configs/coarse_stage_configs*. The separation of the dataset into training and validation samples is defined in *data/HaN_MICCAI2015_Dataset/test_data_split.csv* 

* `python train_configs.py --config script_configs/train_config.txt --gpu 0 --stage coarse`

##### Coarse Stage Inference
 
For an intermediate evaluation of the coarse stage the segmentation results on the validation set are generated and evaluated using:

* `python infer_trained_configs.py --config script_configs/inference_config.txt --gpu 0 --stage coarse --checkpoint 50000 --splitfile data/HaN_MICCAI2015_Dataset/test_data_split.csv`
* `python script/rescale.py --datasetpath coarse_stage --checkpoint 50000`
* `python script/collect_results.py --modeldir coarse_stage --gtdir data/HaN_MICCAI2015_Dataset/full_dataset --resultfile coarse_res.csv --checkpoint 50000 --useplastimatch`
* `python script/evaluate_results.py --resultfile coarse_res.csv`

Finally, the results of the coarse stage produced by the different configurations can be found in *coarse_res_evaluated.csv* and *coarse_res_evaluated_organwise.csv*

To use the coarse network to generate the foreground mask used by the sampler of the fine stage, the pre-trained model of the best coarse configuration available in this repository *(coarse_stage/half_e-3_48-8_dice_1024s)* or any other model of the coarse stage can be used.

* `python infer_trained_configs.py --config script_configs/inference_config.txt --gpu 0 --stage coarse --checkpoint 100000 --splitfile data/HaN_MICCAI2015_Dataset/infer_all_for_fine_stage.csv`
* `python script/rescale.py --datasetpath coarse_stage --checkpoint 100000`
* `python script/create_maskfrom_labels.py --data coarse_stage/half_e-3_48-8_dice_1024s/output/100000 --outdir data/HaN_MICCAI2015_Dataset/mask_fine_stage` 

...


# NiftyNet

<img src="https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/raw/master/niftynet-logo.png" width="263" height="155">

[![build status](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/badges/dev/build.svg)](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/commits/dev)
[![coverage report](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/badges/dev/coverage.svg)](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/commits/dev)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/LICENSE)
[![PyPI version](https://badge.fury.io/py/NiftyNet.svg)](https://badge.fury.io/py/NiftyNet)

NiftyNet is a [TensorFlow][tf]-based open-source convolutional neural networks (CNN) platform for research in medical image analysis and image-guided therapy.
NiftyNet's modular structure is designed for sharing networks and pre-trained models.
Using this modular structure you can:

* Get started with established pre-trained networks using built-in tools
* Adapt existing networks to your imaging data
* Quickly build new solutions to your own image analysis problems

NiftyNet is a consortium of research groups (WEISS -- [Wellcome EPSRC Centre for Interventional and Surgical Sciences][weiss], CMIC -- [Centre for Medical Image Computing][cmic], HIG -- High-dimensional Imaging Group), where WEISS acts as the consortium lead.


### Features

NiftyNet currently supports medical image segmentation and generative adversarial networks.
**NiftyNet is not intended for clinical use**.
Other features of NiftyNet include:

* Easy-to-customise interfaces of network components
* Sharing networks and pretrained models
* Support for 2-D, 2.5-D, 3-D, 4-D inputs*
* Efficient discriminative training with multiple-GPU support
* Implementation of recent networks (HighRes3DNet, 3D U-net, V-net, DeepMedic)
* Comprehensive evaluation metrics for medical image segmentation

 <sup>*2.5-D: volumetric images processed as a stack of 2D slices;
4-D: co-registered multi-modal 3D volumes</sup>

NiftyNet release notes are available [here][changelog].

[changelog]: CHANGELOG.md


### Installation

1. Please install the appropriate [TensorFlow][tf] package*:
   * [`pip install tensorflow-gpu==1.3`][tf-pypi-gpu] for TensorFlow with GPU support
   * [`pip install tensorflow==1.3`][tf-pypi] for CPU-only TensorFlow
1. [`pip install niftynet`](https://pypi.org/project/NiftyNet/)

 <sup>*All other NiftyNet dependencies are installed automatically as part of the pip installation process.</sup>

[tf-pypi-gpu]: https://pypi.org/project/tensorflow-gpu/
[tf-pypi]: https://pypi.org/project/tensorflow/


### Documentation
The API reference and how-to guides are available on [Read the Docs][rtd-niftynet].

[rtd-niftynet]: http://niftynet.rtfd.io/

### Useful links

[NiftyNet website][niftynet-io]

[NiftyNet source code on CmicLab][niftynet-cmiclab]

[NiftyNet source code mirror on GitHub][niftynet-github]

[Model zoo repository][niftynet-zoo]

NiftyNet mailing list: [nifty-net@live.ucl.ac.uk][ml-niftynet]

[Stack Overflow](https://stackoverflow.com/questions/tagged/niftynet) for general questions

[niftynet-io]: http://niftynet.io/
[niftynet-cmiclab]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet
[niftynet-github]: https://github.com/NifTK/NiftyNet
[niftynet-zoo]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer/blob/master/model_zoo.md
[ml-niftynet]: mailto:nifty-net@live.ucl.ac.uk


### Citing NiftyNet

If you use NiftyNet in your work, please cite [Gibson and Li, et al. 2017][preprint]:

* E. Gibson\*, W. Li\*, C. Sudre, L. Fidon, D. I. Shakir, G. Wang, Z. Eaton-Rosen, R. Gray, T. Doel, Y. Hu, T. Whyntie, P. Nachev, M. Modat, D. C. Barratt, S. Ourselin, M. J. Cardoso\^ and T. Vercauteren\^ (2017)
[NiftyNet: a deep-learning platform for medical imaging. arXiv: 1709.03485][preprint]


BibTeX entry:

```
@InProceedings{niftynet17,
  author = {Eli Gibson and Wenqi Li and Carole Sudre and Lucas Fidon and Dzhoshkun I. Shakir and Guotai Wang and Zach Eaton-Rosen and Robert Gray and Tom Doel and Yipeng Hu and Tom Whyntie and Parashkev Nachev and Marc Modat and Dean C. Barratt and Sebastien Ourselin and M. Jorge Cardoso and Tom Vercauteren},
  title = {NiftyNet: a deep-learning platform for medical imaging},
  year = {2017},
  eprint = {1709.03485},
  eprintclass = {cs.CV},
  eprinttype = {arXiv},
}
```
The NiftyNet platform originated in software developed for [Li, et al. 2017][ipmi2017]:

* Li W., Wang G., Fidon L., Ourselin S., Cardoso M.J., Vercauteren T. (2017)
[On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task.][ipmi2017]
In: Niethammer M. et al. (eds) Information Processing in Medical Imaging. IPMI 2017.
Lecture Notes in Computer Science, vol 10265. Springer, Cham.
DOI: [10.1007/978-3-319-59050-9_28][ipmi2017]


[ipmi2017]: http://doi.org/10.1007/978-3-319-59050-9_28
[preprint]: http://arxiv.org/abs/1709.03485


### Licensing and Copyright

Copyright 2018 University College London and the NiftyNet Contributors.
NiftyNet is released under the Apache License, Version 2.0. Please see the LICENSE file for details.

### Acknowledgements

This project is grateful for the support from the [Wellcome Trust][wt], the [Engineering and Physical Sciences Research Council (EPSRC)][epsrc], the [National Institute for Health Research (NIHR)][nihr], the [Department of Health (DoH)][doh], [Cancer Research UK][cruk], [University College London (UCL)][ucl], the [Science and Engineering South Consortium (SES)][ses], the [STFC Rutherford-Appleton Laboratory][ral], and [NVIDIA][nvidia].

[cmic]: http://cmic.cs.ucl.ac.uk
[ucl]: http://www.ucl.ac.uk
[cruk]: https://www.cancerresearchuk.org
[tf]: https://www.tensorflow.org/
[weiss]: http://www.ucl.ac.uk/weiss
[wt]: https://wellcome.ac.uk/
[epsrc]: https://www.epsrc.ac.uk/
[nihr]: https://www.nihr.ac.uk/
[doh]: https://www.gov.uk/government/organisations/department-of-health
[ses]: https://www.ses.ac.uk/
[ral]: http://www.stfc.ac.uk/about-us/where-we-work/rutherford-appleton-laboratory/
[nvidia]: http://www.nvidia.com
[nifty]: https://github.com/NifTK/NiftyNet
[dataset]: http://www.imagenglab.com/newsite/pddca/

