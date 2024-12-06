import json
import numpy as np

file = json.load(open('data_unigram150_with_accent_thresh_4_5.json', 'r'))['utts']

keys = list(file.keys())

probs = np.zeros((5, 5))

for i in range(len(keys)):
    probs[file[keys[i]]['conv_accent']] += np.array(file[keys[i]]['probs'])

probs = probs/np.sum(probs, axis=1)[:, None]

print(np.round(probs*100))
# print(np.sum(probs, axis=1))