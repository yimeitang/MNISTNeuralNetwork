# MNISTNeuralNetwork

### 1. This model allows you to choose 
  (1)Different early stopping criteria <br>
  (2)Different cost functions: a.Quadratic; b.Cross-entropy; c.Log-likelihood; <br>
  (3)SGD or Momentum <br>
  (4)No regularization or L2 Rgularization<br>
  (5)Different transfer(activation) functions: a.tanh; b.softmax; c.ReLU;<br>
  
### 2. The model uses better initial weights and minibatch shuffling, it will return learned network and accuracy/costs for your train set ,validation set and test set.

### 3. Parameters:
The parameters in the function are :<br>
(1) inputs: a matrix with a column for each example, and a row for each input feature.<br>
(2) targets: a matrix with a column for each example, and a row for each output feature.<br>
(3) split: how the data will be divided into training set, test set and validation set.<br>
(4) nodeLayers: a vector with the number of nodes in each layer (including the input and output layers). Important: Your code should not assume that there are just three layers of nodes. It should work with a network of any size.<br>
(5) numEpochs: (scalar) desired number of epochs to run.<br>
(6) batchSize: (scalar) number of instances in a mini-batch.<br>
(7) eta: (scalar) learning rate.<br>
(8) costFunOption: 0 stands for quadradic, 1 stands for cross entropy , 2 stands for log likelihood.<br>
(9) actFunOption: 0 stands for sigmoid, 1 stands for tanh, 2 stands for softmax, 3 stands for ReLU.<br>
(10) momentum: the value of momentum.<br>
(11) lambda: the value of lambda.<br>

### 4. Outputs:
(Early Stopping: In my code, I set the training to terminate early before all epochs have run when the difference between the accuracy score of training set and validation set is larger or equal to 0.15. The reason is that if the difference is larger or equal than 0.15, it means that the model is overfitting and result will not be well generalized.)
![testcases](https://user-images.githubusercontent.com/27776652/32146229-fa2cb2f2-bca1-11e7-8c90-3ac60935db4e.PNG)
![tc1](https://user-images.githubusercontent.com/27776652/32146148-3504e968-bca1-11e7-8f65-3388084858b0.PNG)
![tc1_plot](https://user-images.githubusercontent.com/27776652/32146149-351466d6-bca1-11e7-8b81-ceac59d6694c.PNG)
![tc2](https://user-images.githubusercontent.com/27776652/32146150-352cab6a-bca1-11e7-87ec-f85279b84b1a.PNG)
![tc2_plot](https://user-images.githubusercontent.com/27776652/32146151-3540d5e0-bca1-11e7-9ca2-83697ab49e9a.PNG)
![tc3](https://user-images.githubusercontent.com/27776652/32146152-356b7926-bca1-11e7-9c93-68888d0ac30c.PNG)
![tc3_plot](https://user-images.githubusercontent.com/27776652/32146153-35810b10-bca1-11e7-83f4-32b1f77a38bd.PNG)
![tc4](https://user-images.githubusercontent.com/27776652/32146154-3597b3ce-bca1-11e7-887a-28b1230dd86e.PNG)
![tc4_plot](https://user-images.githubusercontent.com/27776652/32146155-35b32924-bca1-11e7-90e6-e43aeedb0f30.PNG)
![tc5](https://user-images.githubusercontent.com/27776652/32146156-36231806-bca1-11e7-9ce9-2bd7ffca7cf3.PNG)
![tc5_plot](https://user-images.githubusercontent.com/27776652/32146157-36465bfe-bca1-11e7-9090-902769b4c831.PNG)
![tc6](https://user-images.githubusercontent.com/27776652/32146158-3664df84-bca1-11e7-8069-fff3e98fce25.PNG)
![tc6_plot](https://user-images.githubusercontent.com/27776652/32146159-3699e828-bca1-11e7-8d69-31c2137fb8be.PNG)
![tc7](https://user-images.githubusercontent.com/27776652/32146160-36d1b51e-bca1-11e7-880f-26207cc9fdef.PNG)
![tc7_plot](https://user-images.githubusercontent.com/27776652/32146161-36f08f48-bca1-11e7-8bfc-5febc021b08e.PNG)
![tc8](https://user-images.githubusercontent.com/27776652/32146162-37163acc-bca1-11e7-88b4-49db3a5629f0.PNG)
![tc8_plot](https://user-images.githubusercontent.com/27776652/32146163-373324ca-bca1-11e7-9edb-493ba422068e.PNG)
![tc9](https://user-images.githubusercontent.com/27776652/32146164-37499214-bca1-11e7-87f1-8796aeb157b5.PNG)
![tc9_plot](https://user-images.githubusercontent.com/27776652/32146165-3764af0e-bca1-11e7-8ef5-a5bca0f987cf.PNG)


