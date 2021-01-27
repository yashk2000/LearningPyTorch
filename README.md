# 🔦 Learning PyTorch

## 🧰 Using pretrained models in pytorch 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashk2000/LearningPyTorch/blob/main/PyTorchPretrained.ipynb)

Link to notebook: [https://github.com/yashk2000/LearningPyTorch/blob/main/PyTorchPretrained.ipynb](https://github.com/yashk2000/LearningPyTorch/blob/main/PyTorchPretrained.ipynb)

- A pretrained network is a model that has already been trained on a dataset.  
- Such networks produce useful results immediately after loading the network parameters.
- Put the network into eval mode for the dropout and batch normalization layers to work properly before making inferences. 
- Generative adversarial networks (GANs) have two parts—the generator and the discriminator—that work together to produce output indistinguishable from
authentic items. 
- CycleGAN uses an architecture that supports converting back and forth between two different classes of images.
- Torch Hub is a standardized way to load models and weights from any project with an appropriate hubconf.py file.

## 😎 Tensors

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashk2000/LearningPyTorch/blob/main/Tensors.ipynb)

Link to notebook: [https://github.com/yashk2000/LearningPyTorch/blob/main/Tensors.ipynb](https://github.com/yashk2000/LearningPyTorch/blob/main/Tensors.ipynb)

- Neural networks transform floating-point representations into other floating-point representations. The starting and ending representations are typically
human interpretable, but the intermediate representations are less so. These floating-point representations are stored in tensors.
- Tensors are multidimensional arrays; they are the basic data structure in PyTorch
- Tensors can be serialized to disk and loaded back.
- All tensor operations in PyTorch can execute on the CPU as well as on the GPU(`tensor.to(device="cuda")`).
- PyTorch uses a trailing underscore to indicate that a function operates in place on a tensor (for example, Tensor.sqrt_ ).

## 📊 Loading data in PyTorch 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashk2000/LearningPyTorch/blob/main/Datasets.ipynb)

Link to notebook: [https://github.com/yashk2000/LearningPyTorch/blob/main/Datasets.ipynb](https://github.com/yashk2000/LearningPyTorch/blob/main/Datasets.ipynb)

- Neural networks require data to be represented as multidimensional numerical tensors, often 32-bit floating-point.
- Images can have one or many channels. The most common are the red-green-blue channels of typical digital photos. Many images have a per-channel bit depth of 8, though 12 and 16 bits per channel are not uncommon. These bit depths can all be stored in a 32-bit floating-point number without loss of precision.
- Single-channel data formats sometimes omit an explicit channel dimension.
- Volumetric data is similar to 2D image data, with the exception of adding a third dimension (depth).
- Converting spreadsheets to tensors can be very straightforward. Categorical and ordinal-valued columns should be handled differently from interval-valued
columns.

| ![img](https://user-images.githubusercontent.com/41234408/106006939-c4281e80-60db-11eb-9fd2-aa36d2314225.png)  |
|---|
| Deciding between using values directly, one hot encoding or embedding |
