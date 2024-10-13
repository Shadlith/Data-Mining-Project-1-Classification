# import all classifier codes
import DT
import KNN
import NaiveBayes
import RF
import SVM
import Neural

import matplotlib.pyplot as plt

# run all classifiers
DT.DT()
KNN.KNN()
NaiveBayes.NaiveBayes()
RF.RF()
SVM.SVM()
Neural.Neural()

# Run each model and store accuracy values
accuracy_values = {
    'DT': DT.DT(),
    'KNN': KNN.KNN(),
    'NaiveBayes': NaiveBayes.NaiveBayes(),
    'RF': RF.RF(),
    'SVM': SVM.SVM(),
    'Neural': Neural.Neural()
}

# Extract accuracy values from dictionary
accuracies = [value for value in accuracy_values.values()]

# Create bar chart
plt.bar(accuracy_values.keys(), accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.show()
