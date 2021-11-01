# Machine learning tasks
Homework assignments for the course Intelligent Systems (SPbU, 7th semester)

### Task 1
**Dataset**: Facebook Comment Volume Dataset.  
Predict how many comments a post will receive. The task involves the implementation of gradient descent and the calculation of metrics for evaluating the quality of the model. Solution steps:  
* normalization of feature values;  
* cross-validation for five folds and linear regression training;  
* R^2 and RMSE calculation.  

Local run: `python main.py <number of data variants>`  
You can also look at the comm_pred.ipynb. 

### Task 3
**Dataset**: https://snap.stanford.edu/data/loc-Gowalla.html.  
Find clusters on the user graph using the Affinity Propagation method. Compare the effectiveness of these clusters in the task of recommending places.  

Local run: `python cluster.py`  

