# Homework: Building a KNN Classifier

K Nearest Neighbor (KNN) is a simple and effective algorithm. In this homework, we will implement a simple KNN classifier and use it classify Iris flower data. 

## K Nearest Neighbor (KNN)

Upon receiving a new ipute data, KNN searches the training data and finds the K samples with the shortest distance (nearest neighbor). The majority category of these K samples is used as a prediction. The steps of KNN are listed below:

1. After receiving a new ipute matrix, calculate a distance matrix.
   Given an input matrix $X$ and training data matrix $Y$, $X \in \mathbb{R}^{m \times d}, Y \in \mathbb{R}^{n \times d}$. The Euclidean distances of input and trainign samples can be calcuated as $(X - Y)^2=X^2 - 2X*Y^T + Y^2$. We will get a $n \times m$ distance matrix.

2. Sort distance matrix in columns, get K nearest neighbors.

3. Count the majority class of the K nearest neighbors. Report them as predictions.

## Train/Test Data

The data are stored in the [data](./data) folder. The labels are saved in the last column.

## Auto-grading

PyTest will be used to test your model. 
