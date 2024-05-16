import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

aus_feats = np.load('/ASR_Kmeans/data/kmeans_codebooks/aus/feats_6/aus_0_1.npy')
can_feats = np.load('/ASR_Kmeans/data/kmeans_codebooks/can/feats_6/can_0_1.npy')
en_feats = np.load('/ASR_Kmeans/data/kmeans_codebooks/en/feats_6/en_0_1.npy')
sco_feats = np.load('/ASR_Kmeans/data/kmeans_codebooks/sco/feats_6/sco_0_1.npy')
us_feats = np.load('/ASR_Kmeans/data/kmeans_codebooks/us/feats_6/us_0_1.npy')



aus_lens = [155, 282, 436, 332, 166, 219, 333]
can_lens = [165, 257, 317, 262, 178, 275, 275]
en_lens = [237, 342, 273, 309, 263, 236, 287]
sco_lens = [204, 203, 186, 314]
us_lens = [212, 257, 305, 231, 146, 232, 368]


#all_accs = np.vstack((aus_feats[1320:1475][0:50], can_feats[:165][0:50], en_feats[1174:1411][0:50], sco_feats[703:907][0:50], us_feats[:212][0:50]))


# can_feats = np.vstack((np.mean(can_feats[:165,:], axis=0), np.mean(can_feats[165:440,:], axis=0), np.mean(can_feats[440:702,:], axis=0), np.mean(can_feats[702:977,:], axis=0), np.mean(can_feats[977:1234,:], axis=0), np.mean(can_feats[1234:1551,:], axis=0), np.mean(can_feats[1551:1729,:], axis=0)))
# en_feats = np.vstack((np.mean(en_feats[:287,:], axis=0), np.mean(en_feats[287:523,:], axis=0), np.mean(en_feats[523:865,:], axis=0), np.mean(en_feats[865:1174,:], axis=0), np.mean(en_feats[1174:1411,:], axis=0), np.mean(en_feats[1411:1684,:], axis=0), np.mean(en_feats[1684:1947,:], axis=0)))
# sco_feats = np.vstack((np.mean(sco_feats[:314,:], axis=0), np.mean(sco_feats[314:517,:], axis=0), np.mean(sco_feats[517:703,:], axis=0), np.mean(sco_feats[703:907,:], axis=0)))
# us_feats = np.vstack((np.mean(us_feats[:212,:], axis=0), np.mean(us_feats[212:580,:], axis=0), np.mean(us_feats[580:811,:], axis=0), np.mean(us_feats[811:1043,:], axis=0), np.mean(us_feats[1043:1348,:], axis=0), np.mean(us_feats[1348:1494,:], axis=0), np.mean(us_feats[1494:1751,:], axis=0)))



aus_feats = np.vstack((np.mean(aus_feats[:155], axis=0), np.mean(aus_feats[155:437], axis=0), np.mean(aus_feats[437:873], axis=0), np.mean(aus_feats[873:1205], axis=0), np.mean(aus_feats[1205:1371], axis=0), np.mean(aus_feats[1371:1590], axis=0), np.mean(aus_feats[1590:1923], axis=0)))
can_feats = np.vstack((np.mean(can_feats[:165], axis=0), np.mean(can_feats[165:422], axis=0), np.mean(can_feats[422:739], axis=0), np.mean(can_feats[739:1001], axis=0), np.mean(can_feats[1001:1179], axis=0), np.mean(can_feats[1179:1454], axis=0), np.mean(can_feats[1454:1729], axis=0)))
en_feats = np.vstack((np.mean(en_feats[:237], axis=0), np.mean(en_feats[237:579], axis=0), np.mean(en_feats[579:852], axis=0), np.mean(en_feats[852:1161], axis=0), np.mean(en_feats[1161:1424], axis=0), np.mean(en_feats[1424:1660], axis=0), np.mean(en_feats[1660:1947], axis=0)))
sco_feats = np.vstack((np.mean(sco_feats[:204], axis=0), np.mean(sco_feats[204:407], axis=0), np.mean(sco_feats[407:593,:], axis=0), np.mean(sco_feats[593:907], axis=0)))
us_feats = np.vstack((np.mean(us_feats[:212], axis=0), np.mean(us_feats[212:469], axis=0), np.mean(us_feats[469:774], axis=0), np.mean(us_feats[774:1005], axis=0), np.mean(us_feats[1005:1151], axis=0), np.mean(us_feats[1151:1383], axis=0), np.mean(us_feats[1383:1751], axis=0)))

# Concatenate the arrays to create a single array for T-SNE
all_accs = np.concatenate((aus_feats, can_feats, en_feats, sco_feats, us_feats), axis=0)

# Apply T-SNE to reduce the dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42,n_iter=10000, metric='kl')
tsne_result = tsne.fit_transform(all_accs)

# Separate the results back into individual arrays
aus_cl = tsne_result[0:7]
can_cl = tsne_result[7:14]
en_cl = tsne_result[14:21]
sco_cl = tsne_result[21:25]
us_cl = tsne_result[25:32]

# aus_cl = tsne_result[:1923]
# can_cl = tsne_result[1923:3652]
# en_cl = tsne_result[3652:5599]
# sco_cl = tsne_result[5599:6506]
# us_cl = tsne_result[6506:]

# aus_cl = tsne_result[:50]
# can_cl = tsne_result[50:100]
# en_cl = tsne_result[100:150]
# sco_cl = tsne_result[150:200]
# us_cl = tsne_result[200:]

# aus_cl = tsne_result[:155]
# can_cl = tsne_result[155:320]
# en_cl = tsne_result[320:557]
# sco_cl = tsne_result[557:761]
# us_cl = tsne_result[761:]

m_size = 50

# # Plot the results
# plt.scatter(aus_cl[:, 0], aus_cl[:, 1], label='aus', alpha=0.7)
# plt.scatter(can_cl[:, 0], can_cl[:, 1], label='can', alpha=0.7)
# plt.scatter(en_cl[:, 0], en_cl[:, 1], label='en', alpha=0.7)
# plt.scatter(sco_cl[:, 0], sco_cl[:, 1], label='sco', alpha=0.7)
# plt.scatter(us_cl[:, 0], us_cl[:, 1], label='us', alpha=0.7)

# Plot the results
plt.scatter(aus_cl[:, 0], aus_cl[:, 1], label='aus', alpha=0.7, s=m_size)
plt.scatter(can_cl[:, 0], can_cl[:, 1], label='can', alpha=0.7, s=m_size)
plt.scatter(en_cl[:, 0], en_cl[:, 1], label='en', alpha=0.7, s=m_size)
plt.scatter(sco_cl[:, 0], sco_cl[:, 1], label='sco', alpha=0.7, s=m_size)
plt.scatter(us_cl[:, 0], us_cl[:, 1], label='us', alpha=0.7, s=m_size)

plt.title('Outputs from 6th layer of HuBERT')
plt.legend()
plt.savefig("codebooks_6.png")




