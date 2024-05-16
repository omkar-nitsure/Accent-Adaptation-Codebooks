file = open("train.tsv")

lines = file.readlines()


for i in range(1, len(lines)):
    lines[i] = lines[i].split("\t")
    lines[i][-1] = lines[i][-1][:-1]

aus = []
can = []
en = []
sco = []
us = []

for i in range(1, len(lines)):
    if(lines[i][-1] == "australia"):
        aus.append("\t".join(lines[i]) + "\n")
    if(lines[i][-1] == "canada"):
        can.append("\t".join(lines[i]) + "\n")
    if(lines[i][-1] == "england"):
        en.append("\t".join(lines[i]) + "\n")
    if(lines[i][-1] == "scotland"):
        sco.append("\t".join(lines[i]) + "\n")
    if(lines[i][-1] == "us"):
        us.append("\t".join(lines[i]) + "\n")

with open("aus.txt", "w") as f:
    for i in range(len(aus)):
        f.writelines(aus[i])
f.close()

with open("can.txt", "w") as f:
    for i in range(len(can)):
        f.writelines(can[i])
f.close()

with open("en.txt", "w") as f:
    for i in range(len(en)):
        f.writelines(en[i])
f.close()

with open("sco.txt", "w") as f:
    for i in range(len(sco)):
        f.writelines(sco[i])
f.close()

with open("us.txt", "w") as f:
    for i in range(len(us)):
        f.writelines(us[i])
f.close()
