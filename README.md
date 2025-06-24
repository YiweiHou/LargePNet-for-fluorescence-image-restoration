# LargePNet-for-fluorescence-image-restoration

This the supplementary code repository of the article "LargePNet: Large-view statistics aggregation enables finer restoration of fluorescence images"

It contains a general fluorescence image resotration model LargePNet, and its extended verions.

Its core idea is to aggregate large-view global information to improve the restoration performance. Its primary technical details are shown in the below figure.

To exert the advantages of LargePNet, it typically requires training images with a resolution of 512x512

![LargePNet](./Image/1.png)

We have explored eight representative fluorescence image restoration tasks，all showing advantages of LargePNet over conventional small-patch-trained networks, as shonw below:

![LargePNet](./Image/2.png)

## 💻 Testing environment
  - Windows 11
  - CUDA 11.8
  - Python 3.10
  - Pytorch 2.0.0
  - NVIDIA GPU (GeForce RTX 4080) 

## 🎨 Datasets and trained models download link
  In the peer-review stage these can only be downloaded by the reviewers using the Private Link provided in the Data availability sectioin of the manuscript. 

  Once published they will be all open-source.
  
## ✨ Usage
1. Environment setup
   
   We recommend using Anaconda to setup a virtual environment for LargePNet. In Anaconda Prompt, for example:
   
   conda create -n LargePNetEnv python=3.10
   
   conda activate LargePNetEnv
   
   Then, install the package listed in requirements.txt in the created virtual environment
   
   We provided jupyter notebook .ipynb files to directly illustrate the training and inference procedure
   
   Jupyter core shall be installed in the virtual environment before using:
   
   conda activate LargePNet
   
   pip install --user ipykernel
   
   python -m ipykernel install --user --name=LargePNetEnv
   
   After install, .ipynb files can be run with web browser or Pycharm Profession Edition
   
3. Training

   We have provided .ipynb files of model training on different restoration tasks.
   
   Users can follow the cell in .ipynb files to run the training code, and only need to change the Data_dir and save_dir according to specific needs.
   
   As an example of training on BioSR single-image super-resolution dataset:
   
   Users can download our dataset file BioSR.zip, unzip it, and placed it in the Data folder
   
   In the Train_SISR.ipynb file, users only need to specify: Data_dir = r'Data\BioSR\ER\Training', then click each cell for training
   
   The default model training saving directory would be: Data_dir +r'\logfile\LargePNet'
   
   The tqdm package is employed to display the training progress and loss variation.
   
5. Inference

   Users can download our pretrained models and place it in the TrainedModel directory, or use self-trained models for inference.
   
   As an example of inferring BioSR single-image super-resolution dataset with self-trained model:
   
   In the Test_SISR.ipynb file, specify:
   
   head_dir = r"Data\BioSR\ER\Testing"
   
   model.load_state_dict(torch.load(r'Data\BioSR\ER\Training\logfile\LargePNet\best_model.pth'))

   Then click each cell for inference
   
   The output .tif files would be placed at: head_dir + r'\output\LargePNet'

## Acknowledgement

  The development of the code refers the codes provided by the following works: 

  1. Ding, X. et al. Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs. In IEEE Conference on Computer Vision and Pattern Recognition (2022).

  2. Jin, L. et al. Deep learning enables structured illumination microscopy with low light levels and enhanced speed. Nature Communications 11 (2020).

  3. Qiao, C. et al. Evaluation and development of deep neural networks for image super-resolution in optical microscopy. Nature Methods 18, 194-202 (2021).

  4. Weigert, M. et al. Content-aware image restoration: pushing the limits of fluorescence microscopy. Nature Methods 15, 1090-1097 (2018).
