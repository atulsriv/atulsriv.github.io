# Introduction to Deep Learning and Stochastic Gradient Descent (SGD)

Recently, I've been exploring deep learning.  In particular, I've been going through Daniel Bourke's PyTorch Course [Learn Pytorch for Deep Learning](learnpytorch.io) and Jeremy Howard's [Practical Deep Learning for Coders](https://course.fast.ai/). So far, I really recommend both of them for someone who is not too knowledgeable about the domain. 

In particular, I am going to be focusing on [Chapter 4](https://nbviewer.org/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb) project of the FastAI course which focuses on Stochastic Gradient Descent. By the end of the article, I want to showcase a full example of how to create a digit classifier using Stochastic Gradient Descent and creating our own neural network!

## The Problem 
![]({{ site.baseurl }}/images/sgd-links/five.png)

How can we classify this image of a 5 programmatically? Easy enough to do in our brains, but let's explore how to do this with some simple deep learning. I'm going to focus on how to do this using **PyTorch**, but then I will also try to do a quick walk through of how to do this using some higher level functionalities offered by **FastAI** 


## The Dataset
The [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset is a convenient dataset that features a collection of images of digits, split up into 60,000 images to be used in a training set, and 10000 to be used in a validation set. 

In order to download this dataset easily, we can use some functionality already built in PyTorch! 

## Prerequisites 
I chose to run this in Google Colab because of free access to a GPU. This makes training on the model we are about to create much faster and easier than running it with a local jupyter notebook on my CPU. It is quite easy to get started with it, check it out here: [Google Colab](https://colab.google/). To configure running on a GPU, you can select the Runtime tab on the top of the screen and select a GPU (you can also configure your python versions here). One drawback of Colab is probably that you are forced to use Google Drive for all of your files, but for this task I didn't find it too annoying.


## Walkthrough
I'm going to do my best go over the underlying code for this tutorial.

```python
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

trainset = datasets.MNIST('train/', download=True, train=True, transform=transforms.ToTensor())
valset = datasets.MNIST('valid/', download=True, train=False, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True)

```
* From `torchvision` we import `datasets`. This gives us access to the MNIST dataset, which is conveniently already split into training and validation data. 
* The parameter of `transform=transfroms.ToTensor()` converts all of this data into a [PyTorch Tensor](https://pytorch.org/docs/stable/tensors.html). This seems like the most used data structure in PyTorch.
* PyTorch [DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) are a wrapper class for Datasets, which provides helper code for processing data samples. We are creating a DataLoader for the training set and the validation set.

```python
dataiter = iter(trainloader)
images, labels = next(dataiter)
```

* **DataLoaders** gives us an iterator. _iter_ and _next_ are functions on python iterators, and we use next to get an item from _trainloader_. 

```python
print(images.shape)
print(labels.shape)

```


---
torch.Size([128, 1, 28, 28])

torch.Size([128])

---

Let's go over the outputs of these prints. The `iter` call gives us an iterator to features and lebels. The call `images.shape` is actually an alias to the `size()` call in PyTorch, and gives us `torch.Size([128,1,28,28])`. The `128` is the batch size specified above when creating the DataLoader. The pictures in the MNIST Database are of size 28x28. Similarly, since we have `128` images per batch, we also have `128` labels. We can plot our images as well!

```python
figure = plt.figure()
for index in range(1, 128):
  plt.subplot(10,13,index)
  plt.axis('off')
  plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
```

![]({{ site.baseurl }}/images/sgd-links/image_plot.png)
)

## Creating a Neural Network
Quite frankly, it took some time to wrap my head around the neural network concept of this classifier. This video does a great job of explaining it, albeit the example is using Keras: https://www.youtube.com/watch?v=viHXPOgSvBo. Tune into minute 8-11. 

This is also another great article: https://towardsdatascience.com/everything-you-need-to-know-about-neural-networks-and-backpropagation-machine-learning-made-easy-e5285bc2be3a

```python
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)
``` 

![]({{ site.baseurl }}/images/sgd-links/nueral-network.png)

We are following this general pattern, going from an input layer, applying some functions (hidden layer), and coming up with an output layer.
The input layer is a flattened version of our image, you can think of this as a 1-D array. `input_size` being 784 is simply just 28x28, from the pixels of the image.


Can you guess why `output_size = 10`?

If you're thinking that there are 10 digits that we need to classify, then yes, you are correct! 

That leaves the hidden layers. I am no expert in how to optimize performance of this layer at this time, but for now, it is generally understood that for most applications, you can get good performance with just 1 layer here. Our example will just use 1.

Some terminology: `ReLu` stands for Rectified Linear Unit. It is a classification of an activation function. This is a good [article](https://towardsdatascience.com/explain-like-im-five-activation-functions-fb5f532dc06c) on gaining a simple understanding of activation functions, and why ReLU  is a good choice for image classification problems. In particular, it is non-linear, and it converts all negative inputs to zero, so the network on activates neurons that have positive sums. This makes sense, because pixels should never go negative. 

`LogSoftMax` is popular in classification problems where there is a probability of certain classifications. In our problem, we are trying to find the probabilities of 0-9 digits. 


```python
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) 
loss = criterion(logps, labels)
```

- Here, we are defining the loss function used to adjust the weights in our model. `NLLLoss()` represents the negative log-likelihood loss.

```python
print('Before first pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)
```

The next piece of code is to ensure to sanity check to make sure our loss function is working. The `backward()` function is updating the weights to train the model on, so it makes sense that before this is called we have weights set to a default of `None`.   

---
Before backward pass: 

 None

After backward pass: 

 tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
---

Finally, we are going to define an optimizer to actually train the model! 

```python
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.8)
epochs = 20
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        #This is where the model learns by backpropagating
        loss.backward()

        #And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
```

When creating a PyTorch `SGD` object, we are passing in a learning rate and the momentum. The learning rate controls how much of the parameters are updated on each iteration. A higher learning rate will cause the model to learn quickly, but it may also cause the model to overshoot the minumum whereas a low learning rate will cause the model to learn more slowly but has a higher chance of converging to the mimumum. Momentum helps to prevent the model from ocsillating around the minumum, by adding a fraction of the previous update to the current update. To learn more about this, I recommend reading chapter 4 in FastAI course. 

Taking a look at our training loop: 
- each iteration, iterate over images and labels of what's in the training set
- flatten the image into a 784 length vector
- `zero_grad` zeroes out the current gradients.
- update the weights of the model using back propogation
- in each iteration, we should see decreases in the loss

---
Epoch 0 - Training loss: 0.09368936425603147

Epoch 1 - Training loss: 0.09149941415198322

Epoch 2 - Training loss: 0.08968762875492893

Epoch 3 - Training loss: 0.08797932287523233

Epoch 4 - Training loss: 0.08586189654796744

Epoch 5 - Training loss: 0.08429176029143558

Epoch 6 - Training loss: 0.08265308933709857

Epoch 7 - Training loss: 0.08082013065293273

Epoch 8 - Training loss: 0.0794231861885359

Epoch 9 - Training loss: 0.0778691039156558

Epoch 10 - Training loss: 0.07628412325896307

Epoch 11 - Training loss: 0.07503617865476273

Epoch 12 - Training loss: 0.07368348187395632

Epoch 13 - Training loss: 0.07220337166985087

Epoch 14 - Training loss: 0.0710970155998016

---

```python
images, labels = next(iter(valloader))

img = images[0].view(1, 784)
with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
```

---
Predicted Digit = 5

---



```python
correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)


    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
```

---
Number Of Images Tested = 10000

Model Accuracy = 0.9739

---

We have a 97.4% accuracy using this model!


## Using FastAI Library Instead

We saw how we can do this classification model using `PyTorch`, now let's also go over how to do this with `FastAI`. This will involve much less code.

```Python
# Preparing Training Datasets
'''
- total of 60000 training images of size 28x28
- each digit has uniform distrubution
'''

train_images_list = get_image_files(path/'training')
train_x_list = [tensor(Image.open(img_path)) for img_path in train_images_list]
train_y_list = [int(img_path.parent.name) for img_path in train_images_list]
train_x = (torch.stack(train_x_list).float()/255).view(-1,28*28)
train_y = tensor(train_y_list).view(-1,1)
```

- Initialize training/validation sets

```python
train_x.shape, train_y.shape
```

---
(torch.Size([60000, 784]), torch.Size([60000, 1]))

---
 - Notice our training set is already flattened to a size of 784.

```python
train_dset = list(zip(train_x, train_y))
```

```python

valid_images_list = get_image_files(path/'testing')
valid_x_list = [tensor(Image.open(img_path)) for img_path in valid_images_list]
valid_y_list = [int(img_path.parent.name) for img_path in valid_images_list]
valid_x = (torch.stack(valid_x_list).float()/255).view(-1,28*28)
valid_y = tensor(valid_y_list).view(-1,1)

valid_x.shape, valid_y.shape
```

---
(torch.Size([10000, 784]), torch.Size([10000, 1]))

---

```python
valid_dset = list(zip(valid_x, valid_y))
```

```python
dls = ImageDataLoaders.from_folder(path, train='training',valid='testing')
learn = cnn_learner(dls, resnet18, pretrained=False,
                    loss_func=F.cross_entropy, metrics=accuracy, n_out=10)
learn.fit_one_cycle(1, 0.1)
```

To get a sense of accuracy using `resnet18` (very popular image classification model), we can just use this code. By passing in `F.cross_entropy` as a loss function, FastAI will create a `cnn_learner` and train for us.

---
epoch	train_loss	valid_loss	accuracy	time

0	     0.090158	0.358463	0.986000	01:32

---

Running this yields a model with 98.6% accuracy!

We can also create our own DataLoader and train the model similar to how we did in PyTorch. 

```python
train_dl = DataLoader(train_dset, batch_size=256)
#valid_dl = DataLoader(valid_dset, batch_size=256)
```

```python
# function to calculate loss
def mnist_loss(pred, actual):
    l = nn.CrossEntropyLoss()
    return l(pred, actual.squeeze())

# function to calculate gradient
def calc_grad(xb, yb, model):
    pred = model(xb)
    loss = mnist_loss(pred, yb)
    loss.backward()
    return loss

# function to define accuracy
def batch_accuracy(pred, actual):
    digit_pred = pred.max(dim=1)[1]
    return (digit_pred==actual.squeeze()).float().mean()

#function to train 1 epoch and print average batch loss
def train_epoch(model):
    batch_loss = []
    for xb,yb in train_dl:
        batch_loss.append(calc_grad(xb, yb, model))
        opt.step()
        opt.zero_grad()
    return tensor(batch_loss).mean()
```

```python
#Optimizer
class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
        
```

```python
simple_net = nn.Sequential(
    nn.Linear(28*28,100),
    nn.ReLU(),
    nn.Linear(100,30),
    nn.ReLU(),
    nn.Linear(30,10)
)
```

```python
batch_accuracy(simple_net(valid_x),valid_y)
```

```python
opt = BasicOptim(simple_net.parameters(), lr=0.003)
```


```python
def train_model(model,epochs):
    print('{:<10}{:<15}{:<15}'.format('Epoch','Training Loss','Validation Accuracy'))
    for i in range(epochs):
        avg_bl = train_epoch(model)
        print('{:<10}{:<15,.2f}{:<15,.2f}'.format(i,avg_bl.item(),batch_accuracy(model(valid_x),valid_y).item()))

```

```python
train_model(simple_net, 500)
``` 

---
Epoch     Training Loss  Validation Accuracy
0         2.29           0.21           
1         2.26           0.35           
2         2.23           0.41           
3         2.18           0.46           
4         2.12           0.49   
.......
     
457       0.09           0.95           
458       0.09           0.95           
459       0.09           0.95           
460       0.09           0.95
---


## Useful Links
This example uses a combination of the PyTorch Library and the FastAI Library. Find the docs here: 

[GitHub Repo Containing Notebook Code Article](https://github.com/atulsriv/digit-classifier)

[PyTorch](https://pytorch.org/docs/stable/search.html?q=get_image_files&check_keywords=yes&area=default)

[FastAI](https://docs.fast.ai/callback.captum.html)


