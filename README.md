# DecisionTree-ML-NLP
Train a model using an ID3 Decision Tree where it can correctly predict which 'whether' or 'weather' to use in an instance.

Yun-Ching (Kenny) Wu

Decision Tree

Python Version 3.10.7

Default Configurations (No Change)

Change input file names if needed on Lines 182-184

Current Input Files: 'hw1.dev.col', 'hw1.test.col', 'hw1.train.col'

Terminal Command:
>main.py


The code does not display learning curve automatically. You must change it manually.
To change percentage of Training data go to Line 192 and change percent variable 

Percent range:(0.1-1.0)

Line 192 Before: percent = 1.0
Line 192 After:  percent = 0.2

The same must be done if you want to limit the height of the tree (Go to Line 193):

Height range: (1-24)

Line 193 Before: max_height = 3
Line 193 After:  max_height = 5


Add more Features by adding more words in Line 199
