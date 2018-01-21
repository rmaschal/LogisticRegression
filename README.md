This is my own simple Logistic Regression classifier, which I intend to run
on the EMNIST dataset.

Originally the intention was to try and make a k class classifier, and view
our LogClassifier as a 2-layer neural net in that way. However, I couldn't find
a good loss function for k classification when the output is a vector. One idea
is average the loss function for binary classification, but I'll save this for
some other time.

Some features I want to support:
1) Regularization (L1, L2)
2) HyperParameter sweep (for learning rate, regularization parameter

For the parameter sweep, I wonder if I can build a utility that can be usable
for other models as well

The end goal for this project should be a class/utilities generic enough 
to work for any labelled dataset, which can be specified

Current Results:
Two Layer Neural Network:
10 iters :  ~72% accuracy
100 iters:  ~88% accuracy

