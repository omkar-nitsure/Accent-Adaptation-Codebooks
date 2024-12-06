import json
import numpy as np

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
cb_thresh = 0.9

names = {'aus':0, 'can':1, 'en':2, 'sco':3, 'us':4}
a = load_json('train_beams.json')
dest = load_json('data_unigram150_with_accent.json')

keys = list(a.keys())

for i in range(len(keys)):
    prob_dist = np.zeros(5)
    b = a[keys[i]]
    beams = []
    for j in range(len(b)):
        if(len(b[j]) != 0):
            beams.append(b[j])

    index = 0

    for j in range(len(beams) - 1, -1, -1):
        vals = []
        for acc in range(5):
            vals.append(np.sum(np.array(beams[j]) == acc))
        val = np.max(np.array(vals))
        val = val/len(beams[j])
        if(val >= cb_thresh):
            if(j == len(beams) - 1):
                conv_acc = int(np.argmax(vals))
            continue
        else:
            if(j == len(beams) - 1):
                conv_acc = int(np.argmax(vals))
            index = j
            break

    if(index == len(beams) - 1):
        for k in range(len(beams[index])):
            prob_dist[beams[index][k]] += 1
    # elif(((len(beams) - 1) - index) < 3):
    #     for j in range(len(beams) - 1, index, -1):
    #         for k in range(len(beams[j])):
    #             prob_dist[beams[j][k]] += 1
    # elif((((len(beams) - 1) - index) > 3) and (((len(beams) - 1) - index) < 5)):
    #     for j in range(len(beams) - 4, index, -1):
    #         for k in range(len(beams[j])):
    #             prob_dist[beams[j][k]] += 1
    
    for j in range(len(beams) - 1, index, -1):
        for k in range(len(beams[j])):
            prob_dist[beams[j][k]] += 1
    dest['utts'][keys[i]]['probs'] = list(prob_dist)
    dest['utts'][keys[i]]['conv_accent'] = conv_acc

json.dump(dest, open('data_unigram150_with_accent_thresh_9_10.json', 'w'), indent=4)
# accs = []


# prob_dist = np.zeros((5, 5))
# totals = np.zeros(5)

# for i in range(len(keys)):
#     accs.append(names[keys[i].split('_')[0]])

# conv = 0

# for i in range(len(keys)):
#     b = a[keys[i]]
#     beams = []
#     for j in range(len(b)):
#         if(len(b[j]) != 0):
#             beams.append(b[j])

#     index = 0

#     for j in range(len(beams) - 1, -1, -1):
#         vals = []
#         for acc in range(5):
#             vals.append(np.sum(np.array(beams[j]) == acc))
#         val = np.max(np.array(vals))
#         val = val/len(beams[j])
#         if(val >= cb_thresh):
#             if(j == len(beams) - 1):
#                 conv_acc = np.argmax(vals)
#                 # conv_acc = accs[i]
#             continue
#         else:
#             if(j == len(beams) - 1):
#                 conv_acc = np.argmax(vals)
#                 # conv_acc = accs[i]
#             index = j
#             break

#     # if(conv_acc == accs[i]):
#     #     conv += 1
#     # conv_acc = accs[i]

#     if(index == len(beams) - 1):
#         for k in range(len(beams[index])):
#             prob_dist[conv_acc][beams[index][k]] += 1
#             totals[conv_acc] += 1
#     elif(((len(beams) - 1) - index) < 3):
#         for j in range(len(beams) - 1, index, -1):
#             for k in range(len(beams[j])):
#                 prob_dist[conv_acc][beams[j][k]] += 1
#                 totals[conv_acc] += 1
#     elif((((len(beams) - 1) - index) > 3) and (((len(beams) - 1) - index) < 5)):
#         for j in range(len(beams) - 4, index, -1):
#             for k in range(len(beams[j])):
#                 prob_dist[conv_acc][beams[j][k]] += 1
#                 totals[conv_acc] += 1
#     else:
#         for j in range(len(beams) - 6, index, -1):
#             for k in range(len(beams[j])):
#                 prob_dist[conv_acc][beams[j][k]] += 1
#                 totals[conv_acc] += 1

# prob_dist = (prob_dist.T / totals).T
# print(prob_dist)