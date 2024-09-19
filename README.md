# BoutRightYOLO

# Installation
Download anaconda
https://www.anaconda.com/download


# Getting Started
Follow the following steps to get started with boutRight:

1. **Clone the repository**

Clone the repository using Git, if you have a git installed in your computer. To download git go to [Downloads](https://git-scm.com/downloads) and choose a git suitable for your operating system. You can learn more about git [here](https://www.freecodecamp.org/news/git-and-github-for-beginners/).
Optionally, you can install git in anaconda prompt by typing the following command:

```
conda install git
```

Once the git download is done, open a terminal or command prompt and run the following command:

```
git clone https://github.com/RuthKassahun/BoutRightYOLO.git
```

If you don't have a git installed, alternatively, you can download the zip file of this repository if you haven't installed git on your local machine.

2. ***Navigate to the Project Folder***
   
Open Anaconda Prompt and navigate to the "boutRight" folder by using the cd command. Replace ***/path/to/boughtRight_yolo*** with the actual path to the "boutRight" folder on your local machine.
```
cd /path/to/boutRight_yolo
```
3. ***Create a Virtual Environment***

Create a virtual environment using the provided environment.yml file. This YAML file specifies the dependencies and environment settings required for the project.

```
conda env create -f boutright_environment.yml
```
4. ***Activate the Environment***

Activate the newly created conda environment 
```
conda activate boutRight
```
Incase you want to do some sanity check if pytorch works with the configured cuda version 
type `python` once you can see these `>>>` type

```import pytorch```

```torch.cuda.get_device_name(0)``` -- This will display the GPU device your cuda is currently accessing

## Proces Dataset (Generate Spectrogram from song file)
place all your song files inside `songs_for_training` folder in the main `boutRight_yolo` folder

1. Prepare a virtual envrinoment for labelImg
   
   1.1 Create a new conda envirnoment for annotating spectograms using `labelImg`. As LabelImg runs in python 3.8
    ```
    conda create -n imageannotation python==3.8
    ```
   1.2 Activate the envirnoment
    ```
    conda activate imageannotation
    ```
    1.3 Install LabelImg
    ```
    pip install labelImg
    ```

3. Update your `data.yaml` configuration file

   Open data.yaml file and change the directory path for both train and val.
   ```
   train : {\path\to\boutRight_yolo}\dataset\images\train
   val :   {\path\to\boutRight_yolo}\dataset\val
   ```
   for example : if you saved your boutRight_yolo folder in a the following path `D:\Code\boutRight_yolo`, you can change it as
   ```
   train :   D:\Code\boutRight_yolo\dataset\images\train
   val :     D:\Code\boutRight_yolo\dataset\images\val
   ```

4. Open boutRight_yolo_train.ipynb jupyter notebook in visual studio code or in localhost
   This will go through all wav files in the given `songs_for_training` folder and create a new folder named `images_for_training` inside your main directory and add all the generated spectrograms.
   Run the first two cells

6. Open LabelImg
   Inside labelImg conda envirnoment type
   ```
   labelImg
   ```
   This will open a small window to annotate the spectrograms

7. Place all your annotation files inside `annotations` folder in the main directory, then run the third cell (Split Dataset section) from the notebook.

## Train Network
8. Run the network - you can adjust img_size (--img), epochs (--epochs)

9. Once the training is done, you can run `val.py` to evaluate the performance of the network in your validaiton dataset.

   
Common known issues-

while training the network you might come across the following RunTimeError 

`NotImplementedError: Could not run 'torchvision::nms' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'torchvision::nms' is only available for these backends: [CPU, Meta, QuantizedCPU, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMeta, Tracer, AutocastCPU, AutocastXPU, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].
`

Fix the issue by uninstalling torch 

```pip uninstall torch torchvision```

then installing torchvision and torchaudio with the proper cuda verison

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116```

## Model Inference
   
   If you want to use the pretrained model and run it for your custom dataset without training, you can use `boutRight_yolo_infer` notebook
   1. Open ```boutRight_yolo_infer.ipynb```
   2. Create a temporary folder called `Temp` and change the directory of ```temp_path``` to your specified directory
      ```
      temp_path = {path/to/temp_folder}
      ```




