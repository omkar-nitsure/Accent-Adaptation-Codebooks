aus = open("val_aus.tsv", "r")
us = open("val_us.tsv", "r")
en = open("val_en.tsv", "r")
sco = open("val_sco.tsv", "r")
can = open("val_can.tsv", "r")

lines_aus = aus.readlines()
lines_us = us.readlines()
lines_can = can.readlines()
lines_en = en.readlines()
lines_sco = sco.readlines()

id_aus = []
id_us = []
id_can = []
id_en = []
id_sco = []

for i in range(1, len(lines_aus)):
    id_aus.append(int(lines_aus[i].split()[0].split("_")[-1][:-4]))

for i in range(1, len(lines_us)):
    id_us.append(int(lines_us[i].split()[0].split("_")[-1][:-4]))

for i in range(1, len(lines_en)):
    id_en.append(int(lines_en[i].split()[0].split("_")[-1][:-4]))

for i in range(1, len(lines_can)):
    id_can.append(int(lines_can[i].split()[0].split("_")[-1][:-4]))

for i in range(1, len(lines_sco)):
    id_sco.append(int(lines_sco[i].split()[0].split("_")[-1][:-4]))

same_ids = []

for i in range(len(id_us)):
    
    for j in range(len(id_aus)):
        if id_us[i] == id_aus[j]:
            same_ids.append(id_us[i])

print(len(same_ids))
