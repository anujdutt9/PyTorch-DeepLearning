# ResNet Implementation in PyTorch

In this notebook, we'll be implementing the ResNet model uisng PyTorch.

Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

## Why ResNet?

We know that as we develop deeper neural networks, we usually face the problem of Vanishing gradients i.e. as we go down the layers of a deep neural network, the gradients during the backpropagation process keeps getting smaller and smaller due to which after some time, some of the neurons might not even fire leading to poor training of the neural network.

This problem has been addressed by using various techniques like Normalizing Initializations, adding Intermediate Normalization layers etc.

Even after using the abovesaid solutions, the training accuracy for the deeper neural netowrks gets saturated after some time. Even adding more layers does not helps with the training.

ResNet architecture was developed to solve this problem.

## Building Block

The ResNet architecture has a building block which is shown as below:

![building_block](https://miro.medium.com/max/1140/1*D0F3UitQ2l5Q0Ak-tjEdJg.png)

Let us understand this block in detail.

### Residual Function Intuition
1. In this block architecture, we have the Input "x" which is usually an image.

2. This is then fed into the first layer, say a "2D Convolution" layer.

3. After that, we add a Batch Normalization Layer and apply ReLU activation.

The output of this activation is represented as below:

```
Residual Function
F(x) := H(x) - x        ..(1)

where,
x: Input to the building block
H(x): Learned underlying mapping
F(x): Residual Function
```

This output looks different from a normal convolutional neural network architecure as here instead of learning about the input "x", we are trying to learn the underlying mapping H(x) using the stacked layers of the building block.

If we hypothesizes that multiple nonlinear layers can asymptotically approximate complicated functions, then it is equivalent to hypothesize that they can asymptotically approximate the residual functions, i.e., H(x) − x.

So rather than expect stacked layers to approximate H(x), we
explicitly let these layers approximate a residual function F(x) := H(x) − x. The original function at the output (before activation) thus becomes F(x)+x.

Hence, the output of this layer is represented by F(x) in equation (1) which represents the residual mapping.

### Output Function Intuition
4. Once we get the residual mapping from the first layer, then this output is used as input to the second layer consisting of a "2D Convolution" operation followed by "Batch Normalization" layer.

5. Now before applying the ReLU activation to the output of the second layer, we add the Input "x" to the output of the second layer. This is represented as follows:

```
Recasting the output to Original Mapping
y = F(x) + x           ..(2)

where,
F(x): Output of First Layer.
x: Input to this building block.
y: Output of the building block.
```

Adding the input to the output of the final layer in the building block before applying the activation function helps us to recast the underlying mapping F(x) in equation (1) to the original mapping represented by y in equation (2).

Since, we add (element wise addition) the original Input as an Identity, it doesn't adds any new parameters to the neural network, as it's not part of the final Weight Matrix, as well as doesn't adds any extra complexity.

## Building Block Formalization

We adopt residual learning to every few stacked layers. The building block is defined as the following:

```
y = F(x, {Wi}) + x     ..(3)

where,
x: Input Vector
y: Output Vector
F(x, {Wi}): Function representing the residual mapping to be learned
```

For example, for the image in the previous section, if we represent the first weight layer by W1 and the second weight layer by W2, the Output Vector can be represented as:

```
y = W2 * ReLu(W1 * x)   ..(4)
```

Additionally, there could be cases where we need to perform dimensionality mapping. For example, going from one layer with 64 filters to another layer with 128 filters, we need to make sure the dimensions of the output of the previous layer matches the input of the next layer.
We know that the final output vector "y" is achieved by the elementwise addition of the learned residual mapping and the identity mapping from the shortcut connection.
To solve the problem of matching the dimensions, we perform a linear projection "Ws" by the shortcut connections to match the dimensions.

Hence, the output vector in the equation (3) is changed to the following:

```
y = F(x, {Wi}) + Wsx    ..(5)

where,
Ws: is usually a 1x1 convolution operation
```

## Bottleneck Block

The basic building block works well for small number of layers in the architecture but when we scale the number of layers upto like 50/101/152 and so on, we use a different variant of the building block for improving the training process.

The Bottleneck Block looks like below:

![Bottleneck Block](https://i.stack.imgur.com/kbiIG.png)

The above figure shows a comparison between the Basic Block and the Bottleneck Block.

In the Bottleneck Block, for each residual function F, we use a stack of 3 layers instead of 2 as shown in the figure above. The three layers are 1×1, 3×3, and 1×1 convolutions, where the 1×1 layers are responsible for reducing and then increasing (restoring) dimensions, leaving the 3×3 layer a bottleneck with smaller input/output dimensions.

So, now that we have covered most of the basics of the ResNet architecure, it's time to implement them.


```python
# Import Dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
```

# Building Block a.k.a Residual Block



```python
# Basic ResNet Block
class ResidualBlock(nn.Module):
  # expansion: Block expansion parameter in order to increase the out_channels if needed
  expansion= 1

  # Constructor
  # The Building Block requires the following:
  # Input Dimension, Output Dimension, Stride
  def __init__(self, input_channels, output_channels, stride=1, dim_change=None):
    super(ResidualBlock, self).__init__()

    # First Layer
    self.conv1 = nn.Conv2d(in_channels= input_channels,
                           out_channels= output_channels,
                           kernel_size= 3,
                           stride= stride,
                           padding=1)
    
    # Batch Normalization 1
    self.bn1 = nn.BatchNorm2d(output_channels)

    # Second Layer
    self.conv2 = nn.Conv2d(in_channels= output_channels,
                           out_channels= output_channels,
                           kernel_size= 3,
                           stride= 1,
                           padding=1)
    
    # Batch Normalization 2
    self.bn2 = nn.BatchNorm2d(output_channels)

    # Dimension Change Flag
    # Dimension change is required for the output of the current block if the 
    # number of channels in the next block are more than the current block.
    self.dim_change = dim_change
    
  # Forward Pass
  def forward(self, x):
    # Residue
    res = x
    # F(x) := H(x) - x
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))

    # Check if a dimension change is required
    if self.dim_change is not None:
      res = self.dim_change(res)
    
    # y = F(x) + x
    out += res
    out = F.relu(out)
    return out
```

# Bottleneck Block


```python
# Bottleneck Block
class BottleNeckBlock(nn.Module):
  # expansion: Block expansion parameter in order to increase the out_channels if needed
  # In the Bottleneck Block, the last 1x1 Conv layer has 4 times the number of channels
  # as compared to the previous layer
  expansion= 4

  # Constructor
  # The Building Block requires the following:
  # Input Dimension, Output Dimension, Stride
  def __init__(self, input_channels, output_channels, stride=1, dim_change=None):
    super(BottleNeckBlock, self).__init__()

    # 1x1 Convolution
    self.conv1 = nn.Conv2d(in_channels= input_channels,
                           out_channels= output_channels,
                           kernel_size= 1,
                           stride= 1)
    
    # Batch Normalization 1
    self.bn1 = nn.BatchNorm2d(output_channels)
    
    # 3x3 Convolution
    self.conv2 = nn.Conv2d(in_channels= output_channels,
                           out_channels= output_channels,
                           kernel_size= 3,
                           stride= stride,
                           padding=1)
    
    # Batch Normalization 2
    self.bn2 = nn.BatchNorm2d(output_channels)

    # 1x1 Convolution
    # Mutiply output channels with expansion to increase the output dimension
    self.conv3 = nn.Conv2d(in_channels= output_channels,
                           out_channels= output_channels * self.expansion,
                           kernel_size= 1)
    
    # Batch Normalization 3
    self.bn3 = nn.BatchNorm2d(output_channels * self.expansion)

    # Dimension Change Flag
    # Dimension change is required for the output of the current block if the 
    # number of channels in the next block are more than the current block.
    self.dim_change = dim_change

  # Forward Pass
  def forward(self, x):
    res = x

    # 1x1 Layer 1: Input [Reduce Size]
    out = F.relu(self.bn1(self.conv1(x)))

    # 3x3 Conv Layer
    out = F.relu(self.bn2(self.conv2(out)))

    # 1x1 Layer 2 : Output [Expand Size]
    out = self.bn3(self.conv3(out))

    # Check if dimension change is required
    if self.dim_change is not None:
      res = self.dim_change(res)
    
    # Add Shortcut connect to get output
    out += res
    out = F.relu(out)
    return out
```

Now that we have the basic blocks, we just need to bring them together to form the network as shown below:

![ResNet34](https://raw.githubusercontent.com/floydhub/imagenet/master/images/resnet.png)

# ResNet Model


```python
# ResNet Architecture
class ResNet(nn.Module):
  def __init__(self, block, num_layers, num_classes=None):
    super(ResNet, self).__init__()

    # Input Layer
    self.input_channels = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(num_features=64)

    # 2nd block Layers
    # buildingBlock: which block to use; Residual Block or Bottleneck Block
    # num_channels: number of channels per layer in the block
    # num_layers: number of blocks per block layer
    self.layer1 = self._layer(buildingBlock=block, num_channels=64, num_layers=num_layers[0], stride=1)

    # 3rd block Layers
    self.layer2 = self._layer(buildingBlock=block, num_channels=128, num_layers=num_layers[1], stride=2)

    # 4th block layers
    self.layer3 = self._layer(buildingBlock=block, num_channels=256, num_layers=num_layers[2], stride=2)

    # 5th block layers
    self.layer4 = self._layer(buildingBlock=block, num_channels=512, num_layers=num_layers[3], stride=2)

    # Avg Pooling layer
    self.avgpool = nn.AvgPool2d(kernel_size= 4, stride= 1)
    
    # Output Layer
    # Fully connected Layer
    self.fc = nn.Linear(in_features= 512 * block.expansion, out_features= num_classes)

  def _layer(self, buildingBlock, num_channels, num_layers, stride=1):
    dim_change = None

    # Check if dimension changes i.e. if the stride of the next block is > 1
    # or if the number of channels in the next block are more than the current block
    if (stride != 1) or (num_channels != self.input_channels * buildingBlock.expansion):
      dim_change = nn.Sequential(
                              # Perform 1x1 Convolution on the Input i.e. x and increase it's dimension to match
                              # the number of channels in the next block input.
                              # Ex. going from block 1 with channels 64 to block 2 with channels 128
                               nn.Conv2d(in_channels= self.input_channels,
                                         out_channels= num_channels * buildingBlock.expansion,
                                         kernel_size= 1,
                                         stride= stride),
                               # Apply Batch Normalization to that
                               nn.BatchNorm2d(num_features= num_channels * buildingBlock.expansion))

    # Form the Number of Block Layers equal to "num_layers" i.e. how many times the
    # selected block is repeatedly stacked
    # Create a Sequential Model with layers of the block
    net_layers = []
    # Input Layer of each Block
    # If the dimenion change is required, it's required at the input layer block
    net_layers.append(buildingBlock(self.input_channels, num_channels, stride=stride, dim_change=dim_change))
    # Update Input Channels
    self.input_channels = num_channels * buildingBlock.expansion

    for i in range(1, num_layers):
      # Append rest of the Blocks in the layer
      net_layers.append(buildingBlock(self.input_channels, num_channels))
      # Update the Input Channels
      self.input_channels = num_channels * buildingBlock.expansion
    
    return nn.Sequential(*net_layers)

  # Forward Pass
  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    
    # Stack up Layers of Blocks
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # Average pool
    x = F.avg_pool2d(x, 4)

    # Convert Output from 3D to 2D
    x = x.view(x.size(0), -1)

    # Get the Output
    x = self.fc(x)

    return x
```


```python
# Define the image transform
# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1].
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
```


```python
# Load Training Data
train_data = torchvision.datasets.CIFAR10(root='./cifar_data/', train=True, download=True, transform=transform)

# Load Test Data
test_data = torchvision.datasets.CIFAR10(root='./cifar_data/', train=False, download=True, transform=transform)
```

    Files already downloaded and verified
    Files already downloaded and verified



```python
# Create Train Data Loader
trainLoader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=3)

# Create Test Data Loader
testLoader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=3)
```


```python
# Define the Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```




    device(type='cuda')



# ResNet Architectures

The below image shows various ResNet model Architectures with their configurations.

![ResNet Architectures](https://neurohive.io/wp-content/uploads/2019/01/resnet-architectures-34-101.png)


```python
# Define ResNet Architectures
# ResNet-8
model = ResNet(block= ResidualBlock,
               num_layers= [2, 2, 1, 1],
               num_classes=10)

# ResNet-18
# model = ResNet(block= ResidualBlock,
#                num_layers= [2, 2, 2, 2],
#                num_classes=10)

# ResNet-34
# model = ResNet(block= ResidualBlock,
#                num_layers= [3, 4, 6, 3],
#                num_classes=10)

# ----- More number of layers call for Bottleneck Block ------
#ResNet-50
# model = ResNet(block= BottleNeckBlock,
#                num_layers= [3, 4, 6, 3],
#                num_classes=10)

# ResNet-101
# model = ResNet(block= BottleNeckBlock,
#                num_layers= [3, 4, 23, 3],
#                num_classes=10)

# ResNet-152
# model = ResNet(block= BottleNeckBlock,
#                num_layers= [3, 8, 36, 3],
#                num_classes=10)
```


```python
# Send Model to Device
model.to(device)
```




    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (layer1): Sequential(
        (0): ResidualBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ResidualBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): ResidualBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (dim_change): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ResidualBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): ResidualBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (dim_change): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (layer4): Sequential(
        (0): ResidualBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (dim_change): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (avgpool): AvgPool2d(kernel_size=4, stride=1, padding=0)
      (fc): Linear(in_features=512, out_features=10, bias=True)
    )




```python
# Loss Criteria
loss_criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(),
                      lr=0.02,
                      momentum=0.9)
```

# Training the Model


```python
# Train the Model
Epochs = 10

test_loss = 0
train_losses, test_losses = [], []
test_accuracy = []

for epoch in range(Epochs):
  running_loss = 0.0

  for i, batch in enumerate(trainLoader, 0):
    # Load data in batches
    images, labels = batch
    images, labels = images.to(device), labels.to(device)

    # Zero the Optimizer Gradient Values
    optimizer.zero_grad()

    # Define Model output
    prediction = model(images)
    
    # Define Loss
    loss = loss_criterion(prediction, labels)
    
    # Do backward Propagation of gradients
    loss.backward()
    
    # Take one step for optimizer
    optimizer.step()
    
    # Capture Loss
    running_loss += loss.item()

    # Print Loss and Accuracy
    if i%100 == 0:
        print("Epoch: {0}\t i: {1}\t Loss: {2}".format(epoch+1, i+1, running_loss/1000))
        # Reset Running Loss
        running_loss = 0.0
  
  correctPredictions = 0
  totalPredictions = 0

  with torch.no_grad():
    for batches in testLoader:
      images, labels = batches
      images,labels = images.to(device), labels.to(device)
      prediction = model(images)
      # Get test loss
      test_loss += loss_criterion(prediction, labels)
      # Get Index of output class with max probability
      _, prediction = torch.max(prediction.data, 1)
      totalPredictions += labels.size(0)
      correctPredictions += (prediction==labels).sum().item()
    
    test_accuracy.append(correctPredictions/totalPredictions)
    train_losses.append(running_loss/len(trainLoader))
    test_losses.append(test_loss/len(testLoader))

    print("\n-----------------------------\n")
    print("Epoch: {0}\t Accuracy: {1}%".format(epoch+1, str((correctPredictions/totalPredictions)*100)))
    print("\n-----------------------------\n")
```

    Epoch: 1	 i: 1	 Loss: 0.002357080936431885
    Epoch: 1	 i: 101	 Loss: 0.1703247915506363
    Epoch: 1	 i: 201	 Loss: 0.13468822705745698
    Epoch: 1	 i: 301	 Loss: 0.11421894830465316
    
    -----------------------------
    
    Epoch: 1	 Accuracy: 66.11%
    
    -----------------------------
    
    Epoch: 2	 i: 1	 Loss: 0.0010275622606277465
    Epoch: 2	 i: 101	 Loss: 0.08872080427408219
    Epoch: 2	 i: 201	 Loss: 0.08167236816883088
    Epoch: 2	 i: 301	 Loss: 0.07209127745032311
    
    -----------------------------
    
    Epoch: 2	 Accuracy: 75.05%
    
    -----------------------------
    
    Epoch: 3	 i: 1	 Loss: 0.0007279384136199952
    Epoch: 3	 i: 101	 Loss: 0.058263260811567304
    Epoch: 3	 i: 201	 Loss: 0.05834160143136978
    Epoch: 3	 i: 301	 Loss: 0.057205986261367796
    
    -----------------------------
    
    Epoch: 3	 Accuracy: 78.93%
    
    -----------------------------
    
    Epoch: 4	 i: 1	 Loss: 0.0003225184977054596
    Epoch: 4	 i: 101	 Loss: 0.0396798504292965
    Epoch: 4	 i: 201	 Loss: 0.04266443145275116
    Epoch: 4	 i: 301	 Loss: 0.04421730849146843
    
    -----------------------------
    
    Epoch: 4	 Accuracy: 81.12%
    
    -----------------------------
    
    Epoch: 5	 i: 1	 Loss: 0.0002388879656791687
    Epoch: 5	 i: 101	 Loss: 0.02750944845378399
    Epoch: 5	 i: 201	 Loss: 0.02926413509249687
    Epoch: 5	 i: 301	 Loss: 0.032470101043581966
    
    -----------------------------
    
    Epoch: 5	 Accuracy: 80.63%
    
    -----------------------------
    
    Epoch: 6	 i: 1	 Loss: 0.00028972426056861876
    Epoch: 6	 i: 101	 Loss: 0.018277334339916705
    Epoch: 6	 i: 201	 Loss: 0.02024198040366173
    Epoch: 6	 i: 301	 Loss: 0.023575541220605373
    
    -----------------------------
    
    Epoch: 6	 Accuracy: 81.92%
    
    -----------------------------
    
    Epoch: 7	 i: 1	 Loss: 0.0001471358835697174
    Epoch: 7	 i: 101	 Loss: 0.011549804981797933
    Epoch: 7	 i: 201	 Loss: 0.011514306277036666
    Epoch: 7	 i: 301	 Loss: 0.013120515167713166
    
    -----------------------------
    
    Epoch: 7	 Accuracy: 81.76%
    
    -----------------------------
    
    Epoch: 8	 i: 1	 Loss: 0.00012120538204908371
    Epoch: 8	 i: 101	 Loss: 0.00782944506406784
    Epoch: 8	 i: 201	 Loss: 0.008254372917115688
    Epoch: 8	 i: 301	 Loss: 0.009547030229121447
    
    -----------------------------
    
    Epoch: 8	 Accuracy: 81.97%
    
    -----------------------------
    
    Epoch: 9	 i: 1	 Loss: 5.2049480378627775e-05
    Epoch: 9	 i: 101	 Loss: 0.0067277482822537425
    Epoch: 9	 i: 201	 Loss: 0.004587210539728403
    Epoch: 9	 i: 301	 Loss: 0.005068981721997261
    
    -----------------------------
    
    Epoch: 9	 Accuracy: 81.77%
    
    -----------------------------
    
    Epoch: 10	 i: 1	 Loss: 9.8869189620018e-05
    Epoch: 10	 i: 101	 Loss: 0.00459636278450489
    Epoch: 10	 i: 201	 Loss: 0.003989343019202352
    Epoch: 10	 i: 301	 Loss: 0.004737277701497078
    
    -----------------------------
    
    Epoch: 10	 Accuracy: 82.06%
    
    -----------------------------
    


# Save Trained Model


```python
# Save Trained Model
torch.save(model, "ResNet-8.pth")
torch.save(model.state_dict(), "ResNet-8-state-dict.pth")
print("Saved ResNet Model...")
```

    Saved ResNet Model...
    
