import numpy as np

text = open("t_text.txt", "r", encoding="utf8")
ids = open("ids.txt", "r")
l_id = ids.readlines()
l_t = text.readlines()
same_ids = []

for i in range(len(l_t)):
    a = [int(l_id[i][:-1])]
    for j in range(len(l_t)):
        if((l_t[i] == l_t[j]) and (i != j)):
            a.append(int(l_id[j][:-1]))
    same_ids.append(a)

len_ids = []
f = []
for i in range(len(same_ids)):
    len_ids.append(len(same_ids[i]))

for i in range(len(len_ids)):
    if(len_ids[i] >= 5 and len_ids[i] <= 15):
        f.append(i)

print(f[0:30])

