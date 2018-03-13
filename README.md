# PyTorch-DeepLearning

***This repository contains the code for Deep Learning Tutorials using PyTorch. All code is written in Python3.6.***

# Requirements

**1. Anaconda [Latest Version]**

**2. PyTorch**

**Windows Installation:**

```
# If your main Python version is not 3.5 or 3.6
conda create -n test python=3.6 numpy pyyaml mkl

# for CPU only packages
conda install -c peterjc123 pytorch-cpu

# for Windows 10 and Windows Server 2016, CUDA 8
conda install -c peterjc123 pytorch

# for Windows 10 and Windows Server 2016, CUDA 9
conda install -c peterjc123 pytorch cuda90

# for Windows 7/8/8.1 and Windows Server 2008/2012, CUDA 8
conda install -c peterjc123 pytorch_legacy
```

**Linux and Mac OS Installation:**

```
http://pytorch.org/
```

***NOTE:*** Instructions to Install PyTorch for Windows taken from https://github.com/peterjc123/pytorch-scripts.

**3. TorchVision**

To install this on Windows:

```
git clone https://github.com/pytorch/vision.git
```

Then using anaconda command prompt, go inside the "vision" folder and do:

```
python setup.py install
```

To install this on Linux/Mac OS:

```
pip3 install torchvision
```

**4. Numpy [+ mkl for Windows]**

```
pip3 install numpy
```


