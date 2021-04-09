# PFN_2019_ml
self-test for Preffered Network Internship 2019
Task definition and datasets can be found [here](https://github.com/pfnet/intern-coding-tasks/tree/master/2019/machine_learning).
# part1
usage
```
git clone https://github.com/pfnet/intern-coding-tasks.git
cp -r intern-coding-tasks/2019/ml ~/ml
cd ~/ml
git clone https://github.com/jennifer66666/PFN_2019_ml.git src

# run utils to test problem 1
cd ~/ml
python3 -m src.utils
```
# part2
```
python3 -m src.problem2
```
It will show the result for problem 2, taking a file pair (_graph.txt and _label.txt) in datasets/train/ as input. Noted that the loss decreases properly when W is initiallized as Gaussian distribution. 
<p align="center">
  <img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/loss_p2_decrease_under0001.png" width="360" height="240" title="Gaussian"/>
</p>
But somehow fails to be optimized when W is initiallized as ones.
<p align="center">
<img src="https://github.com/jennifer66666/PFN_2019_ml/blob/master/loss_nodcrease_ones.png" width="360" height="240" title="Ones"/>
</p>