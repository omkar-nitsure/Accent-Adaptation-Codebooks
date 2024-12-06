import json
import os
import numpy as np

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)['utts']
    
def load_log(path):
    with open(path, 'r') as f:
        l = f.readlines()[11:-2]
    f.close()

    lines = []
    for i in range(len(l)):
        if(l[i][0] == '{'):
            lines.append(l[i])
    return lines
    
    
# acc_files = sorted(os.listdir('dev_split25utt'))
# log_files = sorted(os.listdir('dev_log'))
acc_files = sorted(os.listdir('split100utt'))
log_files = sorted(os.listdir('train_log'))

file = {}

accent_id = np.zeros(5, dtype=int)
names = ['aus', 'can', 'en', 'sco', 'us']

for i in range(len(log_files)):
    acc_file = acc_files[i]
    log_file = log_files[i]
    
    acc = load_json('split100utt/' + acc_file)
    log = load_log('train_log/' + log_file)

    acc_keys = list(acc.keys())

    for j in range(len(log)):

        b = log[j].split(': ')[-1][1:-3].split(',')
        accent = int(acc[acc_keys[j]]['accent'])
        accent_id[accent] += 1

        beams = []
        temp_beam = []
        # print(b)
        # exit()
        # print(b)
        for i in range(len(b)):
            if((b[i][-1] == ']') and (b[i][-2] != '[')):
                temp_beam.append(int(b[i][-2]))
                beams.append(temp_beam)
                temp_beam = []
            elif(b[i][-1] != ']'):
                temp_beam.append(int(b[i][-1]))

        file.update({f"{acc_keys[j]}": beams})


json.dump(file, open('train_beams.json', 'w'), indent=4)

# file_dest = json.load(open('dev/data_unigram150_with_accent.json', 'r'))['utts']
# file_src = json.load(open('valid_beams.json', 'r'))

# src_keys = list(file_src.keys())

# for i in range(len(src_keys)):
#     file_dest[src_keys[i]]['beams'] = "hello world"

# json.dump(file_dest, open('data_unigram150_with_accent_final.json', 'w'), indent=4)