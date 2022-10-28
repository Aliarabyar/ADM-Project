# ADM-Project
Algorithmics for Data Mining Project - pr04
 
Deliverable 4: Comparing the results of Stacking methods of two distinct datasets

We can think of stacking as the process of combining various machine learning models. For
ensemble models, there are several different techniques that can be used, including stacking,
boosting, and bagging.
Because stacking focuses on examining the space of several models used to solve the same problem,
it differs significantly from these ensemble methods.
The basic idea behind this approach is to solve a machine learning problem utilizing various model
types that can learn to a certain amount rather than the entire problem field.
As we mentioned, an ensemble machine learning algorithm is called stacking or stacked generalization.
It learns the most effective way to combine the predictions from two or more base machine learning
models using a meta-learning method.
To clarify more and in order to compare bagging and boosting methods with stacking, we
should mention that, first, stacking is often considered heterogeneous weak learners (various learning
methods are mixed), whereas bagging and boosting assume mainly homogeneous weak learners.
Second, bagging and boosting combine weak learners using deterministic algorithms, whereas stacking
learns to combine the basic models using a meta-model.
A concept diagram of stacking (From Graham Harrison’s website [1]) illustrated in Figure 1.

![image](https://user-images.githubusercontent.com/75095078/198701794-f70c2bb0-7ec8-4e54-b719-fd560c92b48b.png)
Figure 1: Concept diagram of stacking: Make predictions for the Testing/Validation of Data [1]

to integrate the predictions of these basic classifications using decision tree (DT), K-Nearest Neighbor
(k-NN), and Random Forest (RF) as the basic classifications.

• k-NN(k-nearest neighbor)
The supervised machine learning technique known as the k-nearest neighbors (k-NN) can be
used to tackle classification and regression issues. It is simple to use and comprehend, but
it has the important problem of becoming noticeably slower as the amount of data in use
increases.

• RF(random forest)
Random forests, also known as random choice forests, are an ensemble learning method for
classification, regression, and other tasks that works by building a large number of decision
trees during training.

• DT(decision tree)
In statistics, data mining, and machine learning, decision tree learning is a supervised learning
approach. In this formalization, inferences about a set of data are made using a classification
or regression decision tree as a predictive model.
