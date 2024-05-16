import joblib
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

all_c = np.zeros(500)
aus_c = np.zeros(500)
can_c = np.zeros(500)
en_c = np.zeros(500)
sco_c = np.zeros(500)
us_c = np.zeros(500)


aus = open("/ASR_Kmeans/data/kmeans_codebooks/aus/labs_6/aus_0_1.km", "r")
can = open("/ASR_Kmeans/data/kmeans_codebooks/can/labs_6/can_0_1.km", "r")
en = open("/ASR_Kmeans/data/kmeans_codebooks/en/labs_6/en_0_1.km", "r")
sco = open("/ASR_Kmeans/data/kmeans_codebooks/sco/labs_6/sco_0_1.km", "r")
us = open("/ASR_Kmeans/data/kmeans_codebooks/us/labs_6/us_0_1.km", "r")

aus = open("/ASR_Kmeans/data/kmeans_codebooks/labs_baseline/aus_0_1.km", "r")
can = open("/ASR_Kmeans/data/kmeans_codebooks/labs_baseline/can_0_1.km", "r")
en = open("/ASR_Kmeans/data/kmeans_codebooks/labs_baseline/en_0_1.km", "r")
sco = open("/ASR_Kmeans/data/kmeans_codebooks/labs_baseline/sco_0_1.km", "r")
us = open("/ASR_Kmeans/data/kmeans_codebooks/labs_baseline/us_0_1.km", "r")



# aus = open("/ASR_Kmeans/data/ablations/labs_baseline/val_aus_acc_0_1.km", "r")
# can = open("/ASR_Kmeans/data/ablations/labs_baseline/val_can_acc_0_1.km", "r")
# en = open("/ASR_Kmeans/data/ablations/labs_baseline/val_en_acc_0_1.km", "r")
# sco = open("/ASR_Kmeans/data/ablations/labs_baseline/val_sco_acc_0_1.km", "r")
# us = open("/ASR_Kmeans/data/ablations/labs_baseline/val_us_acc_0_1.km", "r")

# aus = open("/ASR_Kmeans/data/ablations/labs_iter4/val_aus_acc_0_1.km", "r")
# can = open("/ASR_Kmeans/data/ablations/labs_iter4/val_can_acc_0_1.km", "r")
# en = open("/ASR_Kmeans/data/ablations/labs_iter4/val_en_acc_0_1.km", "r")
# sco = open("/ASR_Kmeans/data/ablations/labs_iter4/val_sco_acc_0_1.km", "r")
# us = open("/ASR_Kmeans/data/ablations/labs_iter4/val_us_acc_0_1.km", "r")


l_aus = aus.readlines()
l_can = can.readlines()
l_en = en.readlines()
l_sco = sco.readlines()
l_us = us.readlines()

for i in range(len(l_aus)):
    l_aus[i] = l_aus[i].split()

for i in range(len(l_aus)):
    for j in range(len(l_aus[i])):
        l_aus[i][j] = int(l_aus[i][j])

for i in range(len(l_can)):
    l_can[i] = l_can[i].split()

for i in range(len(l_can)):
    for j in range(len(l_can[i])):
        l_can[i][j] = int(l_can[i][j])

for i in range(len(l_en)):
    l_en[i] = l_en[i].split()

for i in range(len(l_en)):
    for j in range(len(l_en[i])):
        l_en[i][j] = int(l_en[i][j])

for i in range(len(l_sco)):
    l_sco[i] = l_sco[i].split()

for i in range(len(l_sco)):
    for j in range(len(l_sco[i])):
        l_sco[i][j] = int(l_sco[i][j])

for i in range(len(l_us)):
    l_us[i] = l_us[i].split()

for i in range(len(l_us)):
    for j in range(len(l_us[i])):
        l_us[i][j] = int(l_us[i][j])




for i in range(len(l_aus)):
    for j in range(len(l_aus[i])):
        aus_c[l_aus[i][j]] += 1
        all_c[l_aus[i][j]] += 1


for i in range(len(l_can)):
    for j in range(len(l_can[i])):
        can_c[l_can[i][j]] += 1
        all_c[l_can[i][j]] += 1


for i in range(len(l_en)):
    for j in range(len(l_en[i])):
        en_c[l_en[i][j]] += 1
        all_c[l_en[i][j]] += 1


for i in range(len(l_sco)):
    for j in range(len(l_sco[i])):
        sco_c[l_sco[i][j]] += 1
        all_c[l_sco[i][j]] += 1


for i in range(len(l_us)):
    for j in range(len(l_us[i])):
        us_c[l_us[i][j]] += 1
        all_c[l_us[i][j]] += 1



# print("no of clusters in AUS ->", np.count_nonzero(aus_c))
# print("no of clusters in CAN ->", np.count_nonzero(can_c))
# print("no of clusters in EN ->", np.count_nonzero(en_c))
# print("no of clusters in SCO ->", np.count_nonzero(sco_c))
# print("no of clusters in US ->", np.count_nonzero(us_c))
# print("no of clusters in all accents ->", np.count_nonzero(all_c))
        
thresh = 0.008

nz_aus = np.zeros(500)
nz_can = np.zeros(500)
nz_en = np.zeros(500)
nz_sco = np.zeros(500)
nz_us = np.zeros(500)
nz_all = np.zeros(500)

for i in range(len(aus_c)):
    if(aus_c[i] > 5):
        nz_aus[i] = 1


for i in range(len(can_c)):
    if(can_c[i] > 5):
        nz_can[i] = 1


for i in range(len(en_c)):
    if(en_c[i] > 5):
        nz_en[i] = 1


for i in range(len(sco_c)):
    if(sco_c[i] > 5):
        nz_sco[i] = 1


for i in range(len(us_c)):
    if(us_c[i] > 5):
        nz_us[i] = 1

# for i in range(len(aus_c)):
#     if(aus_c[i] > int(thresh*np.sum(aus_c))):
#         nz_aus[i] = 1


# for i in range(len(can_c)):
#     if(can_c[i] > int(thresh*np.sum(can_c))):
#         nz_can[i] = 1


# for i in range(len(en_c)):
#     if(en_c[i] > int(thresh*np.sum(en_c))):
#         nz_en[i] = 1


# for i in range(len(sco_c)):
#     if(sco_c[i] > int(thresh*np.sum(sco_c))):
#         nz_sco[i] = 1


# for i in range(len(us_c)):
#     if(us_c[i] > int(thresh*np.sum(us_c))):
#         nz_us[i] = 1

# print(np.sum(all_c))

mat = np.vstack((nz_aus, nz_can, nz_en, nz_sco, nz_us))
plt.imshow(mat, cmap='viridis', aspect='auto')

# Add colorbar to the right of the heatmap
plt.colorbar(label='Values')
plt.savefig("dummy.png")


aus_cmn = np.zeros(5)
can_cmn = np.zeros(5)
en_cmn = np.zeros(5)
sco_cmn = np.zeros(5)
us_cmn = np.zeros(5)

for i in range(500):
    if(nz_aus[i] != 1):
        # if((nz_can[i] == 1)):
        #     aus_cmn[1] += 1
        # if((nz_en[i] == 1)):
        #     aus_cmn[2] += 1
        # if((nz_sco[i] == 1)):
        #     aus_cmn[3] += 1
        # if((nz_us[i] == 1)):
        #     aus_cmn[4] += 1
        if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
            aus_cmn[0] += 1

    if(nz_can[i] != 1):
        # if((nz_aus[i] == 1)):
        #     can_cmn[0] += 1
        # if((nz_en[i] == 1)):
        #     can_cmn[2] += 1
        # if((nz_sco[i] == 1)):
        #     can_cmn[3] += 1
        # if((nz_us[i] == 1)):
        #     can_cmn[4] += 1
        if((nz_aus[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
            can_cmn[1] += 1

    if(nz_en[i] != 1):
        # if((nz_aus[i] == 1)):
        #     en_cmn[0] += 1
        # if((nz_can[i] == 1)):
        #     en_cmn[1] += 1
        # if((nz_sco[i] == 1)):
        #     en_cmn[3] += 1
        # if((nz_us[i] == 1)):
        #     en_cmn[4] += 1
        if((nz_can[i] == 1) and (nz_aus[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
            en_cmn[2] += 1

    if(nz_sco[i] != 1):
        # if((nz_aus[i] == 1)):
        #     sco_cmn[0] += 1
        # if((nz_can[i] == 1)):
        #     sco_cmn[1] += 1
        # if((nz_en[i] == 1)):
        #     sco_cmn[2] += 1
        # if((nz_us[i] == 1)):
        #     sco_cmn[4] += 1
        if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_aus[i] == 1) and (nz_us[i] == 1)):
            sco_cmn[3] += 1

    if(nz_us[i] != 1):
        # if((nz_aus[i] == 1)):
        #     us_cmn[0] += 1
        # if((nz_can[i] == 1)):
        #     us_cmn[1] += 1
        # if((nz_en[i] == 1)):
        #     us_cmn[2] += 1
        # if((nz_sco[i] == 1)):
        #     us_cmn[3] += 1
        if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_aus[i] == 1)):
            us_cmn[4] += 1


# for i in range(500):
#     if(nz_aus[i] == 1):
#         if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
#             aus_cmn[0] += 1
#         if((nz_can[i] != 1) and (nz_en[i] == 1) and (nz_sco[i] != 1) and (nz_us[i] == 1)):
#             aus_cmn[0] += 1
#         if((nz_can[i] != 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] != 1)):
#             aus_cmn[0] += 1
#         if((nz_can[i] == 1) and (nz_en[i] != 1) and (nz_sco[i] != 1) and (nz_us[i] == 1)):
#             aus_cmn[0] += 1
#         if((nz_can[i] == 1) and (nz_en[i] != 1) and (nz_sco[i] == 1) and (nz_us[i] != 1)):
#             aus_cmn[0] += 1
#         if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] != 1) and (nz_us[i] != 1)):
#             aus_cmn[0] += 1

#     if(nz_aus[i] != 1):
#         if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] != 1)):
#             aus_cmn[0] += 1
#         if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] != 1) and (nz_us[i] == 1)):
#             aus_cmn[0] += 1
#         if((nz_can[i] == 1) and (nz_en[i] != 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
#             aus_cmn[0] += 1
#         if((nz_can[i] != 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
#             aus_cmn[0] += 1
#         # if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
#         #     aus_cmn[0] += 1

#     # if(nz_can[i] == 1):
#     #     if((nz_aus[i] == 1) and (nz_en[i] != 1) and (nz_sco[i] != 1) and (nz_us[i] != 1)):
#     #         can_cmn[0] += 1
#     #     if((nz_aus[i] != 1) and (nz_en[i] == 1) and (nz_sco[i] != 1) and (nz_us[i] != 1)):
#     #         can_cmn[2] += 1
#     #     if((nz_aus[i] != 1) and (nz_en[i] != 1) and (nz_sco[i] == 1) and (nz_us[i] != 1)):
#     #         can_cmn[3] += 1
#     #     if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_sco[i] != 1) and (nz_us[i] == 1)):
#     #         can_cmn[4] += 1
#     #     # if((nz_aus[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
#     #     #     can_cmn[1] += 1

#     # if(nz_en[i] == 1):
#     #     if((nz_aus[i] == 1) and (nz_can[i] != 1) and (nz_sco[i] != 1) and (nz_us[i] != 1)):
#     #         en_cmn[0] += 1
#     #     if((nz_aus[i] != 1) and (nz_can[i] == 1) and (nz_sco[i] != 1) and (nz_us[i] != 1)):
#     #         en_cmn[1] += 1
#     #     if((nz_aus[i] != 1) and (nz_can[i] != 1) and (nz_sco[i] == 1) and (nz_us[i] != 1)):
#     #         en_cmn[3] += 1
#     #     if((nz_can[i] != 1) and (nz_aus[i] != 1) and (nz_sco[i] != 1) and (nz_us[i] == 1)):
#     #         en_cmn[4] += 1
#     #     # if((nz_can[i] == 1) and (nz_aus[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
#     #     #     en_cmn[2] += 1

#     # if(nz_sco[i] == 1):
#     #     if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_aus[i] == 1) and (nz_us[i] != 1)):
#     #         sco_cmn[0] += 1
#     #     if((nz_can[i] == 1) and (nz_en[i] != 1) and (nz_aus[i] != 1) and (nz_us[i] != 1)):
#     #         sco_cmn[1] += 1
#     #     if((nz_can[i] != 1) and (nz_en[i] == 1) and (nz_aus[i] != 1) and (nz_us[i] != 1)):
#     #         sco_cmn[2] += 1
#     #     if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_aus[i] != 1) and (nz_us[i] == 1)):
#     #         sco_cmn[4] += 1
#     #     # if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_aus[i] == 1) and (nz_us[i] == 1)):
#     #     #     sco_cmn[3] += 1

#     # if(nz_us[i] == 1):
#     #     if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_sco[i] != 1) and (nz_aus[i] == 1)):
#     #         us_cmn[0] += 1
#     #     if((nz_can[i] == 1) and (nz_en[i] != 1) and (nz_sco[i] != 1) and (nz_aus[i] != 1)):
#     #         us_cmn[1] += 1
#     #     if((nz_can[i] != 1) and (nz_en[i] == 1) and (nz_sco[i] != 1) and (nz_aus[i] != 1)):
#     #         us_cmn[2] += 1
#     #     if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_sco[i] == 1) and (nz_aus[i] != 1)):
#     #         us_cmn[3] += 1
#     #     # if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_aus[i] == 1)):
#     #     #     us_cmn[4] += 1

# for i in range(500):
#     if(nz_aus[i] == 1):
#         if((nz_can[i] == 1) and (nz_en[i] != 1) and (nz_sco[i] != 1) and (nz_us[i] != 1)):
#             aus_cmn[0] += 1
#         if((nz_can[i] != 1) and (nz_en[i] == 1) and (nz_sco[i] != 1) and (nz_us[i] != 1)):
#             aus_cmn[0] += 1
#         if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_sco[i] == 1) and (nz_us[i] != 1)):
#             aus_cmn[0] += 1
#         if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_sco[i] != 1) and (nz_us[i] == 1)):
#             aus_cmn[0] += 1
#         # if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
#         #     aus_cmn[0] += 1

#     if(nz_can[i] == 1):
#         # if((nz_aus[i] == 1) and (nz_en[i] != 1) and (nz_sco[i] != 1) and (nz_us[i] != 1)):
#         #     can_cmn[0] += 1
#         if((nz_aus[i] != 1) and (nz_en[i] == 1) and (nz_sco[i] != 1) and (nz_us[i] != 1)):
#             can_cmn[0] += 1
#         if((nz_aus[i] != 1) and (nz_en[i] != 1) and (nz_sco[i] == 1) and (nz_us[i] != 1)):
#             can_cmn[0] += 1
#         if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_sco[i] != 1) and (nz_us[i] == 1)):
#             can_cmn[0] += 1
#         # if((nz_aus[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
#         #     can_cmn[1] += 1

#     if(nz_en[i] == 1):
#         # if((nz_aus[i] == 1) and (nz_can[i] != 1) and (nz_sco[i] != 1) and (nz_us[i] != 1)):
#         #     en_cmn[0] += 1
#         # if((nz_aus[i] != 1) and (nz_can[i] == 1) and (nz_sco[i] != 1) and (nz_us[i] != 1)):
#         #     en_cmn[0] += 1
#         if((nz_aus[i] != 1) and (nz_can[i] != 1) and (nz_sco[i] == 1) and (nz_us[i] != 1)):
#             en_cmn[0] += 1
#         if((nz_can[i] != 1) and (nz_aus[i] != 1) and (nz_sco[i] != 1) and (nz_us[i] == 1)):
#             en_cmn[0] += 1
#         # if((nz_can[i] == 1) and (nz_aus[i] == 1) and (nz_sco[i] == 1) and (nz_us[i] == 1)):
#         #     en_cmn[2] += 1

#     if(nz_sco[i] == 1):
#         # if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_aus[i] == 1) and (nz_us[i] != 1)):
#         #     sco_cmn[0] += 1
#         # if((nz_can[i] == 1) and (nz_en[i] != 1) and (nz_aus[i] != 1) and (nz_us[i] != 1)):
#         #     sco_cmn[0] += 1
#         # if((nz_can[i] != 1) and (nz_en[i] == 1) and (nz_aus[i] != 1) and (nz_us[i] != 1)):
#         #     sco_cmn[0] += 1
#         if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_aus[i] != 1) and (nz_us[i] == 1)):
#             sco_cmn[0] += 1
#         # if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_aus[i] == 1) and (nz_us[i] == 1)):
#         #     sco_cmn[3] += 1

#     # if(nz_us[i] == 1):
#     #     if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_sco[i] != 1) and (nz_aus[i] == 1)):
#     #         us_cmn[0] += 1
#     #     if((nz_can[i] == 1) and (nz_en[i] != 1) and (nz_sco[i] != 1) and (nz_aus[i] != 1)):
#     #         us_cmn[0] += 1
#     #     if((nz_can[i] != 1) and (nz_en[i] == 1) and (nz_sco[i] != 1) and (nz_aus[i] != 1)):
#     #         us_cmn[0] += 1
#     #     if((nz_can[i] != 1) and (nz_en[i] != 1) and (nz_sco[i] == 1) and (nz_aus[i] != 1)):
#     #         us_cmn[0] += 1
#         # if((nz_can[i] == 1) and (nz_en[i] == 1) and (nz_sco[i] == 1) and (nz_aus[i] == 1)):
#         #     us_cmn[4] += 1
            

# aus_cmn[0] = np.count_nonzero(nz_aus)
# can_cmn[1] = np.count_nonzero(nz_can)
# en_cmn[2] = np.count_nonzero(nz_en)
# sco_cmn[3] = np.count_nonzero(nz_sco)
# us_cmn[4] = np.count_nonzero(nz_us)
            
print("AUS ->", aus_cmn)
print("CAN ->", can_cmn)
print("EN ->", en_cmn)
print("SCO ->", sco_cmn)
print("US ->", us_cmn)
