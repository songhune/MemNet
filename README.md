# MemN2N

Implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895) with sklearn-like interface using Tensorflow. Tasks are from the [bAbl](http://arxiv.org/abs/1502.05698) dataset.

## songhune edited

In order to make my thesis, I've been changing the original code for experiment, if there is any inconvience including licence issues(though I think it would be only used for academical reasons), please contact me with songhune@ajou.ac.kr. Thank you!

### Get Started

```
git clone git@github.com:songhune/MemNet.git

mkdir ./memn2n/data/
cd ./memn2n/data/
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar xzvf ./tasks_1-20_v1-2.tar.gz

cd ../
python single.py
```

### Examples

Running a [single bAbI task](./single.py)

Running a [joint model on all bAbI tasks](./joint.py)

These files are also a good example of usage.

### Requirements

* tensorflow 1.3(or up-to-date)
* scikit-learn 0.18.2
* six 1.11.0

### Single Task Results
songhune edited: on typing


### Notes

Single task results are from 10 repeated trails of the single task model accross all 20 tasks with different random initializations. The performance of the model with the lowest validation accuracy for each task is shown in the table above.

Joint training results are from 10 repeated trails of the joint model accross all tasks. The performance of the single model whose validation accuracy passed the most tasks (>= 0.95) is shown in the table above (joint_scores_run2.csv). The scores from all 10 runs are located in the results/ directory.
