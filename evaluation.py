ground_truth = {
    "resume_1": {"Programming": ["Python", "Java"]},
    "resume_2": {"Data Science": ["Pandas", "NumPy"]},
}

# Assume you’ve already extracted this from your model:
predicted = {
    "resume_1": {"Programming": ["Python", "C++"]},
    "resume_2": {"Data Science": ["Pandas", "TensorFlow"]},
}

from sklearn.metrics import precision_score, recall_score, f1_score

def flatten(skills_dict):
    return [skill.lower() for skills in skills_dict.values() for skill in skills]

y_true = []
y_pred = []

for resume_id in ground_truth:
    true_skills = set(flatten(ground_truth[resume_id]))
    pred_skills = set(flatten(predicted[resume_id]))

    all_skills = list(true_skills.union(pred_skills))
    y_true_binary = [skill in true_skills for skill in all_skills]
    y_pred_binary = [skill in pred_skills for skill in all_skills]

    y_true.extend(y_true_binary)
    y_pred.extend(y_pred_binary)

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1-score:", f1_score(y_true, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

# Example metrics
metrics = {
    'Precision': 0.78,
    'Recall': 0.65,
    'F1-Score': 0.71
}

# Plot
plt.figure(figsize=(6, 4))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='coolwarm')
plt.title("Skill Detection Evaluation Metrics")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.show()

from sklearn.metrics import confusion_matrix
import numpy as np

true_labels = ['Data Science', 'Graphic Design', 'Data Science']
pred_labels = ['Data Science', 'Graphic Design', 'Software Development']

labels = ["Data Science", "Graphic Design", "Software Development"]

cm = confusion_matrix(true_labels, pred_labels, labels=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Career Recommendation")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

accuracy = {
    "Top-1 Accuracy": 0.60,
    "Top-3 Accuracy": 0.85
}

plt.figure(figsize=(6, 4))
sns.barplot(x=list(accuracy.keys()), y=list(accuracy.values()), palette='viridis')
plt.title("Career Recommendation Accuracy")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Sample data: Ground truth vs. predicted labels (ranked)
true_labels = [
    "Data Science", "Graphic Design", "Cybersecurity",
    "Software Development", "Healthcare"
]

predicted_ranked_lists = [
    ["Data Science", "AI", "Software Development"],
    ["UI/UX Design", "Graphic Design", "Content Writing"],
    ["Cybersecurity", "IT", "Software Development"],
    ["AI", "Software Development", "Data Science"],
    ["Education", "Healthcare", "Business Management"]
]

def evaluate_recommendations(true_labels, predicted_ranked_lists):
    top1_hits = 0
    top3_hits = 0
    reciprocal_ranks = []

    for true, predicted in zip(true_labels, predicted_ranked_lists):
        if true == predicted[0]:
            top1_hits += 1
        if true in predicted[:3]:
            top3_hits += 1
        if true in predicted:
            rank = predicted.index(true) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    total = len(true_labels)
    top1_accuracy = top1_hits / total
    top3_accuracy = top3_hits / total
    mrr = sum(reciprocal_ranks) / total

    return {
        "Top-1 Accuracy": round(top1_accuracy, 2),
        "Top-3 Accuracy": round(top3_accuracy, 2),
        "Mean Reciprocal Rank (MRR)": round(mrr, 2)
    }

# Evaluate
metrics = evaluate_recommendations(true_labels, predicted_ranked_lists)
print(metrics)

# Plotting
plt.figure(figsize=(8, 5))
sns.barplot(
    x=list(metrics.keys()),
    y=list(metrics.values()),
    hue=list(metrics.keys()),
    palette='viridis',
    legend=False
)
plt.title("Career Recommendation Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.05)

# Add value labels on bars
for i, val in enumerate(metrics.values()):
    plt.text(i, val + 0.02, f"{val:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

