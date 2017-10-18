.. NiftyNet documentation master file, created by
   sphinx-quickstart on Wed Aug 30 14:13:50 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


NiftyNet
========

NiftyNet is a `TensorFlow`_-based open-source convolutional neural networks (CNN) platform for research in medical image analysis and image-guided therapy.
NiftyNet's modular structure is designed for sharing networks and pre-trained models.
NiftyNet is a consortium of research groups (WEISS -- `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, CMIC -- `Centre for Medical Image Computing`_, HIG -- High-dimensional Imaging Group), where WEISS acts as the consortium lead.


Getting started
---------------

Using NiftyNet's modular structure you can:

* Get started with established pre-trained networks using built-in tools
* Adapt existing networks to your imaging data
* Quickly build new solutions to your own image analysis problems

Please see the `NiftyNet source code repository`_ for a detailed list of features and installation instructions.


Examples
^^^^^^^^

We are working to provide examples here showing how NiftyNet can be used and adapted to different image analysis problems.
In the mean time please see the `NiftyNet demos`_ and `network (re-)implementations`_.

.. _`NiftyNet demos`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/tree/dev/demos
.. _`network (re-)implementations`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/tree/dev/niftynet/network


API reference
^^^^^^^^^^^^^

Please see the :ref:`modindex`.


Useful links
^^^^^^^^^^^^

:ref:`genindex`

:ref:`search`

`NiftyNet website`_

`NiftyNet source code on CmicLab`_

`NiftyNet source code mirror on GitHub`_

NiftyNet mailing list: nifty-net@live.ucl.ac.uk


.. _`NiftyNet website`: http://niftynet.io/
.. _`NiftyNet source code on CmicLab`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet
.. _`NiftyNet source code mirror on GitHub`: https://github.com/NifTK/NiftyNet


Citing NiftyNet
---------------

If you use NiftyNet in your work, please cite `Gibson and Li et. al. 2017`_:

  E. Gibson\*, W. Li\*, C. Sudre, L. Fidon, D. I. Shakir, G. Wang, Z. Eaton-Rosen, R. Gray, T. Doel, Y. Hu, T. Whyntie, P. Nachev, M. Modat, D. C. Barratt, S. Ourselin, M. J. Cardoso\^ and T. Vercauteren\^ (2017)
  `NiftyNet: a deep-learning platform for medical imaging. arXiv: 1709.03485`_

BibTeX entry:

.. code-block:: bibtex

  @InProceedings{niftynet17,
    author = {Eli Gibson and Wenqi Li and Carole Sudre and Lucas Fidon and Dzhoshkun I. Shakir and Guotai Wang and Zach Eaton-Rosen and Robert Gray and Tom Doel and Yipeng Hu and Tom Whyntie and Parashkev Nachev and Marc Modat and Dean C. Barratt and Sebastien Ourselin and M. Jorge Cardoso and Tom Vercauteren},
    title = {NiftyNet: a deep-learning platform for medical imaging},
    year = {2017},
    eprint = {1709.03485},
    eprintclass = {cs.CV},
    eprinttype = {arXiv},
  }

The NiftyNet platform originated in software developed for `Li et. al. 2017`_:

  Li W., Wang G., Fidon L., Ourselin S., Cardoso M.J., Vercauteren T. (2017)
  `On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task.`_
  In: Niethammer M. et al. (eds) Information Processing in Medical Imaging. IPMI 2017.
  Lecture Notes in Computer Science, vol 10265. Springer, Cham. DOI: `10.1007/978-3-319-59050-9_28`_

.. _`NiftyNet: a deep-learning platform for medical imaging. arXiv: 1709.03485`: http://arxiv.org/abs/1709.03485
.. _`Gibson and Li et. al. 2017`: http://arxiv.org/abs/1709.03485
.. _`Li et. al. 2017`: http://doi.org/10.1007/978-3-319-59050-9_28
.. _`On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task.`: http://doi.org/10.1007/978-3-319-59050-9_28
.. _`10.1007/978-3-319-59050-9_28`: http://doi.org/10.1007/978-3-319-59050-9_28


Licensing and copyright
-----------------------

Copyright 2017 University College London and the NiftyNet Contributors.
NiftyNet is released under the Apache License, Version 2.0.
Please see the LICENSE file in the `NiftyNet source code repository`_ for details.


Acknowledgements
----------------

This project is grateful for the support from the `Wellcome Trust`_, the `Engineering and Physical Sciences Research Council (EPSRC)`_, the `National Institute for Health Research (NIHR)`_, the `Department of Health (DoH)`_, `University College London (UCL)`_, the `Science and Engineering South Consortium (SES)`_, the `STFC Rutherford-Appleton Laboratory`_, and `NVIDIA`_.

.. _`TensorFlow`: https://www.tensorflow.org/
.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`NiftyNet source code repository`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet
.. _`Centre for Medical Image Computing`: http://cmic.cs.ucl.ac.uk/
.. _`Centre for Medical Image Computing (CMIC)`: http://cmic.cs.ucl.ac.uk/
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome Trust`: https://wellcome.ac.uk/
.. _`Engineering and Physical Sciences Research Council (EPSRC)`: https://www.epsrc.ac.uk/
.. _`National Institute for Health Research (NIHR)`: https://www.nihr.ac.uk/
.. _`Department of Health (DoH)`: https://www.gov.uk/government/organisations/department-of-health
.. _`Science and Engineering South Consortium (SES)`: https://www.ses.ac.uk/
.. _`STFC Rutherford-Appleton Laboratory`: http://www.stfc.ac.uk/about-us/where-we-work/rutherford-appleton-laboratory/
.. _`NVIDIA`: http://www.nvidia.com


.. toctree::
   :maxdepth: 4
   :caption: Contents:
