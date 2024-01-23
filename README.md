# -Comparative-Analysis-of-Artificial-Neural-Network-Optimization-Algorithms-on-CIFAR-10
Description:<br>
This GitHub project has successfully completed a comparative analysis focusing on experimentally evaluating the performance of optimization algorithms. It utilized a self-designed Multi-Layer Perceptron (MLP) classifier to classify images from the CIFAR-10 dataset, which encompasses multiple classes. The primary objective of the experimental study was to assess the effectiveness of optimization algorithms by individually employing the following optimizers:<br>

RMSprop<br>
SGD (Stochastic Gradient Descent) without momentum<br>
Adam<br>
Adagrad<br>
For all conducted experiments, the learning rates of the optimizers were sequentially set as:<br>

Learning_rate = 1e-2<br>
Learning_rate = 1e-6<br>
The number of epochs for all experimental setups was consistently fixed at 100. The architecture of the MLP network was meticulously determined, taking into account the input neuron count proportional to the image dimensions. The output neuron count was aligned with the total number of classes in the dataset. Additionally, at least one hidden layer was incorporated, with the number of neurons in the hidden layer being a ratio of choice. The flexibility to decide on activation functions between layers was exercised, and all choices were thoughtfully justified.<br>

After each experimental run, comprehensive visualizations were generated to illustrate the changes in train accuracy, train loss, validation accuracy, and validation loss over epochs. Following the testing phase, the project successfully represented the performance metrics using accuracy and a confusion matrix. All results and findings have been compiled into a detailed report.<br>
All implementation codes developed  by using PyTorch.
