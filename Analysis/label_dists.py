import joblib
import numpy as np
import matplotlib.pyplot as plt


def find_dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
    # return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


all = open("aus_overall.txt", "r")
acc = open("kmeans_aus_labels.txt", "r")

lines_all = all.readlines()
lines_acc = acc.readlines()

for i in range(len(lines_all)):
    lines_all[i] = lines_all[i].split()
    lines_acc[i] = lines_acc[i].split()


km_acc = joblib.load("kmeans_aus.sav")
km_all = joblib.load("kmeans_all.sav")

C_acc = km_acc.cluster_centers_
C_all = km_all.cluster_centers_

# C_acc = C_acc / np.sqrt(np.sum((C_acc) ** 2))
# C_all = C_all / np.sqrt(np.sum((C_all) ** 2))

dists = []

for i in range(len(lines_all)):

    l_all = lines_all[i]
    l_acc = lines_acc[i]
    dist = []

    for j in range(len(l_all)):
        dist.append(find_dist(C_acc[int(l_acc[j])], C_all[int(l_all[j])]))

    dists.append(dist)

hist_y = []

for i in range(len(dists)):
    for j in range(len(dists[i])):
        hist_y.append(dists[i][j])

print(np.mean(hist_y))

plt.hist(hist_y)
plt.xlabel("distance between cluster labels for australia")
plt.ylabel("frequency")
plt.legend()
plt.title("Distances between clusters(unit vecs) for acc wise and overall kmeans")
plt.savefig("aus_vecs.png")
plt.show()
