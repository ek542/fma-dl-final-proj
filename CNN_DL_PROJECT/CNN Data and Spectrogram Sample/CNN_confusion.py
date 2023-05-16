import seaborn as sns
import matplotlib.pyplot as plt

confusion_matrix = [
    [125,   8,   1,  17,  22,   7,   7,   4],
    [ 11,  97,  10,  16,  30,  13,   5,  14],
    [  1,  14, 124,   1,  28,  15,  15,   5],
    [ 16,   4,   0, 176,   7,   2,   6,   4],
    [  7,  11,   9,   4, 139,   3,   9,   9],
    [ 10,   4,  14,  20,   5, 127,   5,   8],
    [ 15,  16,  23,  23,  16,  12,  76,  28],
    [  6,   9,   0,   5,  17,   6,  14, 145]
]

genres = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="cool", xticklabels=genres, yticklabels=genres)
plt.xticks(rotation=45)

plt.title("DenseNet121 Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()