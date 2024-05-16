data = open("train.tsv", "r", encoding="utf8")
lines = data.readlines()

lines = lines[1:]

ids = []
text = []
accents = []

for i in range(len(lines)):
    lines[i] = lines[i].split()[1:]
    ids.append(int(lines[i][0].split("_")[-1][:-4]))

data = open("train.tsv", "r", encoding="utf8")
lines = data.readlines()

lines = lines[1:]

for i in range(len(lines)):
    accents.append(lines[i].split()[2:][-2])
    text.append(str(" ".join(lines[i].split()[2:-2])))

file = open('t_ids.txt','w')
for i in range(len(ids)):
	file.write(str(ids[i])+"\n")
file.close()

file = open('t_accents.txt','w')
for i in range(len(accents)):
	file.write(accents[i]+"\n")
file.close()

file = open('t_text.txt', "w")
for i in range(len(text)):
	file.write(text[i]+"\n")
file.close()
