---
draft: false
title: Learning a bit advanced Pytorch
date: 2024-09-07
---

### Training Loop demystified
* **Forward Pass**: Compute predictions (The code where you pass the input to the model to get the output)
* **Loss Calculation**: Compute the loss (The code where you calculate the loss between the predicted output and the target output)
* **Backward Pass**: Compute the gradients for all the parameters where we have requires_grad = True and is stored in the .grad attribute of the parameter (The code where you calculate the gradients of the loss with respect to the model parameters). In the loss.backward() function, it does not update the .grad attribute of the parameter. It just computes the gradient and stores it in the .grad attribute of the parameter.
* **Parameter Update aka optimization**: Update the parameters of the model for which we have .grad attribute / requires grad attributes (The code where you update the parameters using the gradients and a learning rate)
* **Zero_grad()**: The default behavior of the .grad attribute is to accumulate the gradients of the loss with respect to the parameters (The .grad is a tensor and it accumulates the grad for all the parameters). So we need to set it to zero before doing the backward pass.


### Training Loop in Pytorch
```python
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):     
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass (calculate the gradients and stored in .grad attribute of the parameter)
        loss.backward()

        # Updates these parameters based on the defined optimizer and learning rate
        optimizer.step()
```



####  The alias codes and removing the abstraction 

`loss.backward()` translates to  
```
for param in model.parameters():
    if param.requires_grad:
        param.grad = Autograd.backward(loss, param)

```

`Optimizer.step()` translates to 
```
for param in model.parameters():
    if param.grad:
        param.data = param.data - learning_rate * param.grad

```

`optimizer.zero_grad()` translates to 
```
for param in model.parameters():
    if param.requires_grad:
        param.grad = None
```



### Convolutional Layers demystified with alias codes 
[3b1b video conv](https://youtu.be/KuXjwB4LzSA)
Nice [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) for visualizing the convolution operation.

`conv2d( kernel_size = 3, stride = 1, padding = 1)` translates to   
```
input_size = (3 , 32, 32)
>>>kernel_size = 3 == (3, 3, 3) 
stride = 1
padding = 1

output_size = (32, 32)

## get blob of the input_tensor , do element wise multiplication with the kernel and sum them up to get the output_tensor

# Define output tensor of desired size
output_tensor = torch.zeros(output_size)

# Iterate over each position in the output tensor


for c in range(input_size[2]):  
    for i in range(output_size[0]-kernel_size[0]+1):
        for j in range(output_size[1]-kernel_size[1]+1):
            # Define the receptive field
            start_i = i * stride - padding
            start_j = j * stride - padding
            end_i = start_i + kernel_size
            end_j = start_j + kernel_size
            
            # Extract the input patch
            input_patch = input_tensor[:, start_i:end_i, start_j:end_j]
            
            # Perform element-wise multiplication with kernel and sum
            output_tensor[i, j] = torch.sum(input_patch * kernel)

# The output_tensor now contains the result of the convolution operation

```


visualizing `ConvTranspose2d` is a bit more complex and tough than the conv2d layers and the reason is less intutive nature behind it. 
Dont think much about it ... just know that it works and gives results.( I dont know much about it too)

For wrorking with image dataset, we need to do understand , how important resolution is, resolution of an image is the no. of pixels that we have in an image  
*2k image : 2048 x 1080 pixels, more pixels better clarity of image and this way the model also learns better
*4k image : 4096 x 2160 pixels, 

Most models are trained on 512 x 512 dataset , and that is one reason its limited, but if your dataset is at 1024 x 1024 , aka upscaled then that leads to better results 

