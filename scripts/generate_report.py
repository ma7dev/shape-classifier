import pickle
import csv
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
PICKLE_PATH = "/nfs/hpc/share/alotaima/mcs/results.pkl"
loaded_obj = None
with open(PICKLE_PATH, "rb") as f:
    loaded_obj = pickle.load(f)
dd = 220
print(loaded_obj.columns[dd:])

print([col for col in loaded_obj.columns.to_list() if "__" not in col])

seqs_name = ["COL", "GRV", "OP4", "SC","STC4"]
idsw_df_gt = loaded_obj.copy()

# combined mean
print(idsw_df_gt["mistakes"].mean())
print(idsw_df_gt["total"].mean())
print(idsw_df_gt["results"].mean())

# # grouping
for seq_name in seqs_name:
    cond_gt = idsw_df_gt.seq.str.contains(seq_name, case=False)
    idsw_df_gt.loc[idsw_df_gt.index[cond_gt],"seq"] = seq_name

seq_df_gt = idsw_df_gt.copy()
seq_df_updated_gt = seq_df_gt.copy()
str_counted = {0: "1.0", 1: ">0.9", 2: ">0.8", 3: ">0.7", 4: ">0.6", 5: ">.5", 6: "<=.5"}
seq_df_updated_gt.loc[(seq_df_gt['results'] == 1.0),"results"] = 0
seq_df_updated_gt.loc[(seq_df_gt['results'] > .9) & (seq_df_gt['results'] < 1.0),"results"] = 1
seq_df_updated_gt.loc[(seq_df_gt['results'] > .8) & (seq_df_gt['results'] <= .9),"results"] = 2
seq_df_updated_gt.loc[(seq_df_gt['results'] > .7) & (seq_df_gt['results'] <= .8),"results"] = 3
seq_df_updated_gt.loc[(seq_df_gt['results'] > .6) & (seq_df_gt['results'] <= .7),"results"] = 4
seq_df_updated_gt.loc[(seq_df_gt['results'] > .5) & (seq_df_gt['results'] <= .6),"results"] = 5
seq_df_updated_gt.loc[(seq_df_gt['results'] <= .5),"results"] = 6
d_gt = seq_df_updated_gt['results'].to_numpy()
counted_gt = seq_df_updated_gt['results'].value_counts().to_dict()
print(counted_gt)
# set 0 for missing bars
for i in range(0,len(list(str_counted.keys()))):
    counted_gt[i] = counted_gt[i] if i in counted_gt.keys() else 0
counted_gt = dict(sorted(counted_gt.items()))
# # plotting
width = 0.35
fig, ax = plt.subplots()
labels = list(str_counted.values())
x = np.arange(len(labels))
det_count = list(counted_gt.values())
# rects1 = ax.bar(x - width/2, gt_count, width, label='GT', color ='#0504aa', alpha=0.7)
rects2 = ax.bar(x , det_count, width, label='total', color ='#aa0422', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.xlabel('Accuracy Range', fontsize = 10)
plt.ylabel('%', fontsize = 10)
plt.title(f'Total Accuracy', fontweight ='bold', fontsize = 12)
plt.legend()
ax.bar_label(rects2, padding=2)
fig.tight_layout()
plt.savefig('total.png')
# # print mean
for seq_name in seqs_name:
    seq_df_gt = idsw_df_gt.loc[idsw_df_gt['seq'] == seq_name].mean()
    print(f'{seq_name}: det={seq_df_gt["results"]}')
for seq_name in seqs_name:
    print(seq_name)
    seq_df_gt = idsw_df_gt.copy().loc[idsw_df_gt['seq'] == seq_name]
    seq_df_updated_gt = seq_df_gt.copy().loc[idsw_df_gt['seq'] == seq_name]
    seq_df_updated_gt.loc[(seq_df_gt['results'] == 1.0),"results"] = 0
    seq_df_updated_gt.loc[(seq_df_gt['results'] > .9) & (seq_df_gt['results'] < 1.0),"results"] = 1
    seq_df_updated_gt.loc[(seq_df_gt['results'] > .8) & (seq_df_gt['results'] <= .9),"results"] = 2
    seq_df_updated_gt.loc[(seq_df_gt['results'] > .7) & (seq_df_gt['results'] <= .8),"results"] = 3
    seq_df_updated_gt.loc[(seq_df_gt['results'] > .6) & (seq_df_gt['results'] <= .7),"results"] = 4
    seq_df_updated_gt.loc[(seq_df_gt['results'] > .5) & (seq_df_gt['results'] <= .6),"results"] = 5
    seq_df_updated_gt.loc[(seq_df_gt['results'] <= .5),"results"] = 6
    d_gt = seq_df_updated_gt['results'].to_numpy()
    counted_gt = seq_df_updated_gt['results'].value_counts().to_dict()
    # set 0 for missing bars
    for i in range(len(list(str_counted.keys()))):
        counted_gt[i] = counted_gt[i] if i in counted_gt.keys() else 0
    counted_gt = dict(sorted(counted_gt.items()))
    print(counted_gt)
    # plotting
    width = 0.35
    fig, ax = plt.subplots()
    labels = list(str_counted.values())
    x = np.arange(len(labels))
    det_count = list(counted_gt.values())
    rects2 = ax.bar(x, det_count, width, label=f'{seq_name}', color ='#aa0422', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xlabel('Accuracy Range', fontsize = 10)
    plt.ylabel('%', fontsize = 10)
    plt.title(f'{seq_name} Accuracy', fontweight ='bold', fontsize = 12)
    plt.legend()
    ax.bar_label(rects2, padding=2)
    fig.tight_layout()
    plt.savefig(f'{seq_name}.png')