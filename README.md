# PFN_2019_ml
Self-test for Preffered Network Internship 2019. <br>
Task definition and datasets can be found [here](https://github.com/pfnet/intern-coding-tasks/tree/master/2019/machine_learning).
# part1
Usage
```
git clone https://github.com/pfnet/intern-coding-tasks.git
cp -r intern-coding-tasks/2019/ml ~/ml
cd ~/ml
git clone https://github.com/jennifer66666/PFN_2019_ml.git src
cd ~/ml

#test problem1
python3 -m src.utils_np
```
# part2
```
python3 -m src.problem2_np
```
It will show the result for problem 2, taking a file pair (_graph.txt and _label.txt) in datasets/train/ as input. Noted that the loss decreases when W is initiallized as Gaussian distribution. 
<p align="center">
  <img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/pics/loss_p2_decrease_under0001.png" width="360" height="240" title="Gaussian"/>
</p>
But somehow fails to be optimized when W is initiallized as ones.
<p align="center">
<img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/pics/loss_nodcrease_ones.png" width="360" height="240" title="Ones"/>
</p>

# part3

```
# turn on momentum SGD with edit problem3_np.py L147 SGD(split,epochs,batch_size,all=2000,momentum=True)
# turn on Adam with edit problem3_np.py L147 SGD(split,epochs,batch_size,all=2000,adam=True)
# momentum and Adam should not be turned on together
python3 -m src.problem3_np
```
It will test problem3 with the hyperparams listed in the task explanation. Loss shows that the training processes properly. Though the average accuracy is not high as 60% in average as claimed in the task paper. Besides, small batch_size helps. The reason may be that there are too few training data, and large batch_size does not support the weights to update enough times (with equal epochs). Notice that the xv_t should not be given back to the graph, otherwise the training fails. <br>
<br>
These are the results from momentum SGD optimizer. (batch_size 10, split 0.8)
<p align="center">
<img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/pics/loss_train_momentum.png" width="360" height="240" title="loss_train"/><img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/pics/acc_train_momentum.png" width="360" height="240" title="acc_train"/>
<img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/pics/loss_val_momentum.png" width="360" height="240" title="loss_val"/><img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/pics/acc_val_momentum.png" width="360" height="240" title="acc_val"/>
</p>
These are the results from adam optimizer. (batch_size 10, split 0.8)
<p align="center">
<img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/pics/loss_train_adam.png" width="360" height="240" title="loss_train"/><img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/pics/acc_train_adam.png" width="360" height="240" title="acc_train"/>
<img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/pics/loss_val_adam.png" width="360" height="240" title="loss_val"/><img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/pics/acc_val_adam.png" width="360" height="240" title="acc_val"/>
</p>
Though numpy does not support GPU, it is much faster than simple loop on CPU. In ~/bakup, there is loop version for vector/matrix, and calculus on them. _np are postfix for numpy version. <br>
<br>

**Bookmark**

A short survey for Adam and implementation details [here](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) 
