# Vibrational modal shape interpolation through convolutional auto encoder
Repository of "Vibrational modal shape interpolation through convolutional auto encoder"

We propose a first data-driven approach to the super-resolution of vibrational modal shapes.  
The estimation of the mode shapes of a vibrating object is a fundamental task in the study of its dynamic behaviour.
To obtain a meaningful reconstruction of the mode shapes, however, a large number of measurement points is usually needed and a method that allows the reduction of the number of sampling points would permit to speed up the process.  
To solve the problem of reconstructing a high resolution modal shape from subsampled data, we propose a convolutional autoencoder with additional max pooluing residual connections.
This solution was inspired by works in the computer vision literature addressing the super-resolution of natural images.  
The architecture was tested on a dataset of rectangular plates of isotropic material, with simply supported boundary conditions, and the reconstruction performance is analyzed in comparison with a similar network and with bicubic interpolation.
The behaviour of the network to noisy input and missing data has been investigated as well.  
Results on the performed tests suggest that the proposed architecture outperformed model based interpolation and it is able to deal with noisy and different scales inputs.  
  
# Proposed architecture
The model proposed in the paper is a convolutional autoencoder, inspired by the solutions to super-resolution problems proposed in the computer-vision literature, and in particular for the super-resolution of natural images.  
It is composed of an encoding phase, in which input mode shapes images are halved in size by the use of aMax Pooling layer after the convolution, while the number of filters is doubled.
In the decoding phase, on the contrary, the compressed representation of the input data are progressively upsampled through nearest- neighbour interpolation, while the number of filters is reduced.  
Skip connections were added between each encoding steps and the correspoding decoding one.  
Moreover, further skip connections, named MaxRe connections, were added in the encoding phase, to achieve a better reconstruction of the amplitude and the position of maxima.
They are obtained by progressively downsampling the input image with Max Pooling, and concatenating it to the output of the correspoding convolutional step.  
A diagram of the proposed architetcure is depicted below.
The grey and green blocks represent the input and output images, respectively.
Each yellow block represents a layer made by two-dimensional convolution, ReLU activation function and batch normalization.
The output of Max Pooling and Upsampling layers are depicted in red and blue, respectively, while the purple blocks represent MaxRe layers.

![alt text](https://github.com/polimi-ispl/modal-shape-interpolation-cnn/blob/master/images/2020-05-05_220557.png)
