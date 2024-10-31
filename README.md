# Training a Machine Learning Model to Predict XOR Gate in C

This is a project I wrote to explore the fundamental concepts of machine learning and classification algorithms. In main.c, I have provided several training sets to train the model on. The model is able to predict, with high accuracy, what output there should be given two binary inputs, simulating a logic gate.

## Training the XOR Gate

The XOR gate is different to train than the AND, OR, and NAND gates. This is mainly because it is impossible to linearly seperate the classifications of an XOR gate when plotted on a 2D graph. In order to overcome this problem, we train a neural network to simulate a boolean logic circuit equivalent to an XOR gate.

Each neuron in the xor.c file contains three floating point numbers, representing two weights and a bias respectively. The XOR model consists of three neurons, labelled or, and, and nand. We randomly initialize our model, and then iterate over the model to train it. On each training iteration, we calculate cost using forward propagation, approximate a gradient using a finite_differences function, and apply that gradient to each weight/bias using the learning rate hyperparameter. The goal is to reduce the cost function at each iteration. The cost function is the measure of how well our model has been trained on the training data. Below is a diagram of the circuit the model aims to construct.
![Circuit Design:](./img/XOR%20gate.png)

At the end of 100 000 iterations, we reduce our cost function substantially, and the model is able to predict with high accuracy what outputs we should have for a given input.

## Running the Model

Run the XOR model by executing the following commands:

```
cd src
make xor
./xor
```

If you wish to train an AND, NAND, or OR gate, navigate to the main.c file as so:

```
cd src
code main.c
```

Edit the assigment `sample_train* train = nand_train;` to whatever training set you wish to train. Then run the model by running:

```
make main
./main
```
