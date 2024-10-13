
import matplotlib.pyplot as plt
import subprocess
import re

# Function to run each classifier
def run_classifier(script_name):
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    output = result.stdout
    
    # Extract accuracy from the output
    match = re.search(r'Accuracy: ([\d.]+)', output)
    if match:
        return float(match.group(1))
    else:
        print(f"Couldn't find accuracy in output of {script_name}")
        return None



# List of classifier scripts
classifiers = [
    ('Decision Tree', 'DT.py'),
    ('K Nearest Neighbor', 'KNN.py'),
    ('Naive Bayes', 'NaiveBayes.py'),
    ('Random Forest', 'RF.py'),
    ('Support Vector Machine', 'SVM.py'),
    ('Neural Network', 'Neural.py')
]

# Run each model and store accuracy values
accuracy_values = {}
for name, script in classifiers:
    accuracy = run_classifier(script)
    if accuracy is not None:
        accuracy_values[name] = accuracy
 
# Extract accuracy values 
models = list(accuracy_values.keys())
accuracies = list(accuracy_values.values())

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(accuracy_values)), accuracies)
plt.xticks(range(len(accuracy_values)), models, rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.tight_layout()
plt.show()

# Print accuracy values
for name, accuracy in accuracy_values.items():
    print(f"{name}: {accuracy}")