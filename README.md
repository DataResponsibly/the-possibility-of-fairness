[![Python](https://img.shields.io/badge/python-3.7-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Fairness Evaluator

## Description
The ``impossibility theorem'' -- which is considered foundational in algorithmic fairness literature -- asserts that there must be trade-offs between common notions of fairness and performance when fitting statistical models, except in two special cases: when the prevalence of the outcome being predicted is equal across groups, or when a perfectly accurate predictor is used. However, theory does not always translate to practice. In this work, we challenge the implications of the impossibility theorem in practical settings. First, we show analytically that, by slightly relaxing the impossibility theorem (to accommodate a \textit{practitioner's} perspective of fairness), it becomes possible to identify a large set of models that satisfy seemingly incompatible fairness constraints. Second, we demonstrate the existence of these models through extensive experiments on five real-world datasets. We conclude by offering tools and guidance for practitioners to understand when -- and to what degree -- fairness along multiple criteria can be achieved. For example, if one allows only a small margin-of-error between metrics, there exists a large set of models simultaneously satisfying \emph{False Negative Rate Parity}, \emph{False Positive Rate Parity}, and \emph{Positive Predictive Value Parity}, even when there is a moderate prevalence difference between groups. This work has an important implication for the community: achieving fairness along multiple metrics for multiple groups (and their intersections) is much more possible than was previously believed.

Paper: [The Possibility of Fairness: Revisiting the Impossibility Theorem in Practice](https://arxiv.org/abs/2302.06347)

Citation: `Andrew Bell, Lucius Bynum, Nazarii Drushchak, Tetiana Zakharchenko, Lucas Rosenblatt, Julia Stoyanovich (2023). The Possibility of Fairness: Revisiting the Impossibility Theorem in Practice. arXiv preprint arXiv:2302.06347.` 

## Git clone

Clone repositroy:
```bash
git clone https://github.com/DataResponsibly/the-possibility-of-fairness.git
```

In the main project folder, use the following command:

```bash
python setup.py install --user
```

From files in the main directory, you should be able to use
```python
from FairnessEvaluator import FairnessEvaluator
```
Note that this does not work for subfolders.


## Install as package

Install package: 
```bash
pip install git+https://github.com/DataResponsibly/the-possibility-of-fairness.git
```
For installation from any branch with name [branch name]:
```bash
pip install git+https://github.com/DataResponsibly/the-possibility-of-fairness.git@[branch name] 
```

## Test code:
```python
import numpy as np 
import pandas as pd
from FairnessEvaluator import FairnessEvaluator

data_raw = pd.read_csv('X.csv', sep=',')
y = data_raw['G3'].to_numpy()
X = data_raw.drop(columns=['Unnamed: 0','G3'])
X['label_value'] = y
X['idx'] = X.index
X = X[X['Medu'] != 0]
X = X.reset_index(drop=True)

fo = FairnessEvaluator(X=sample_df,
                       label='label',
                       protected_attrs=['sex'],
                       metrics_to_plot=['prec','fpr','fnr'],
                       fairness_bounds=[0.8,1.2],
                       fully_constrained=False,
                       #intersectionality=None,
                       precision_ub = 0.7
                    )    
fo.evaluate()
```
