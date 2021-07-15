import os
import sys
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_path = r'D:\Neuro Publication\All_models_meshed\calc_arrays.npz'
sns.set_context("paper")

npz_arrays = np.load(base_path, allow_pickle=True)

sum_values_df = npz_arrays['sum_values_df'].tolist()
focal_values = npz_arrays['focal_values'].tolist()

fig, ax = plt.subplots()
bars = np.arange(len(focal_values.keys()))
ax.bar(bars, [fc_val['Thal_Both'] for fc_val in focal_values.values()], color ='maroon', width = 0.4, label='Thalamus')
ax.bar(bars, [fc_val['Thal_Right'] for fc_val in focal_values.values()], color ='red', width = 0.4, label='Thalamus Right')

ax.bar(bars + 0.4, [fc_val['Hipp_Both'] for fc_val in focal_values.values()], color ='blue', width = 0.4, label='Hippocampus')
ax.bar(bars + 0.4, [fc_val['Hipp_Right'] for fc_val in focal_values.values()], color ='cyan', width = 0.4, label='Hippocampus Right')

ax.set_xlabel("Models")
ax.set_ylabel("ROI/Rest")
ax.set_xticks(np.arange(0, len(focal_values.keys()), 5) + 0.4/2)
ax.set_xticks(bars + 0.4/2, minor=True)
# ax.set_xticklabels(list(focal_values.keys()))
ax.set_xticklabels(np.arange(1, len(focal_values.keys()) + 1, 5).astype(str))
ax.set_title("Focality")
ax.legend()
plt.grid(True, ls = '--', alpha = 0.5)

fig.tight_layout()
plt.show()


df = pd.DataFrame(sum_values_df).dropna()
df = df.transpose()
#df.to_csv(r'D:\Neuro Publication\Preliminary_Models\sum_over_areas.csv')
corr_df = df.corr(method='pearson')

ticks_minor = np.arange(1.5, len(df.keys()))
ticks_major = np.arange(0.5, len(df.keys()), 5)
labels = np.arange(1, len(df.keys()), 5)
plt.figure()
plt.tight_layout()
sns.set_context('paper', font_scale=1.4)

pl = sns.heatmap(corr_df, annot=False, cmap='coolwarm', robust=True, square=True)
pl.set_xticks(ticks_major)
pl.set_xticks(ticks_minor, minor=True)
pl.set_xticklabels(labels, rotation=0)
pl.set_yticks(ticks_major)
pl.set_yticks(ticks_minor, minor=True)
pl.set_yticklabels(labels, rotation=0)
# pl.set_title("Correlation of area stimulation across models")

plt.show()
