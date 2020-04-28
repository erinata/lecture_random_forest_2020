import kfold_template

import pandas

from sklearn import tree

dataset = pandas.read_csv("dataset.csv")

target = dataset.iloc[:,30].values
data = dataset.iloc[:,0:30].values

# print(target)
# print(data)

machine = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)

r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(3, data, target, machine, 1, 1)

print(r2_scores)
print(accuracy_scores)
for i in confusion_matrices:
	print(i)
