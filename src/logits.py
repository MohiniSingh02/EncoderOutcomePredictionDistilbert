import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchmetrics.classification import MultilabelPrecisionRecallCurve


def get_results_dict(predictions, which='5'):
    mpr = MultilabelPrecisionRecallCurve(num_labels=len(targets.T))

    results_dict = list()
    fpr, tpr, thresholds = mpr(predictions, targets)

    count = 0
    for icd10, p, r, t in zip(all_labels, fpr, tpr, thresholds):
        row = dict()

        f1_score = 2 * p * r / (p + r)
        f1_score = torch.nan_to_num(f1_score, 0)
        ix = torch.argmax(f1_score)
        fscore = f1_score[ix]

        row['icd10'] = icd10
        row['#samples'] = label_dist.loc[icd10]
        row[f'f1_{which}'] = fscore.item()
        row[f'precision_{which}'] = p[ix].item()
        row[f'recall_{which}'] = r[ix].item()

        if t.shape == torch.Size([]):
            row[f'threshold_{which}'] = t.item()
            # print(f'ICD-10 {icd10} with {label_dist.loc[icd10]} #samples Best Threshold={t.item():.2f} Precision={p[ix]:.2f} Recall={r[ix]:.2f} F1={fscore:.2f}')
        else:
            row[f'threshold_{which}'] = t[ix].item()
            # print(f'ICD-10 {icd10} with {label_dist.loc[icd10]} #samples Best Threshold={t[ix]:.2f} Precision={p[ix].item():.2f} Recall={r[ix].item():.2f} F1={fscore:.2f}')

        results_dict.append(row)
    return results_dict


results_dict_5p = get_results_dict(predictions_5p, which='5p')
results_dict_1p = get_results_dict(predictions_1p, which='1p')
results_dict_1p_1280h = get_results_dict(predictions_1p_1280h, which='1p_1280h')

results_df_5p = pd.DataFrame.from_records(results_dict_5p)
results_df_1p = pd.DataFrame.from_records(results_dict_1p)
results_df_1p_1280h = pd.DataFrame.from_records(results_dict_1p_1280h)

results_df_5p.index = results_df_5p.icd10
results_df_1p.index = results_df_1p.icd10
results_df_1p_1280h.index = results_df_1p_1280h.icd10
results_df_5p.to_csv('/pvc/shared/continual/data/results_df_5p.csv', index=False)
results_df_1p.to_csv('/pvc/shared/continual/data/results_df_1p.csv', index=False)
results_df_1p_1280h.to_csv('/pvc/shared/continual/data/results_df_1p_1280h.csv', index=False)

full_results = results_df_5p.join(results_df_1p[[col for col in results_df_1p.columns if col not in ['icd10', '#samples']]])
full_results = full_results.join(results_df_1p_1280h[[col for col in results_df_1p_1280h.columns if col not in ['icd10', '#samples']]])
better_f1_df = full_results[full_results.f1_5p > full_results.f1_1p]
worse_f1_df = full_results[full_results.f1_5p < full_results.f1_1p]


def plot_metrics(df, step=0.2):
    # bins = [6, 11, 51, 101, 1001, 1e6]
    bins = np.array([10 ** i for i in np.arange(1, 5, step)])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric_name in enumerate(['precision', 'recall', 'f1'][::-1]):
        results_1p = list()
        results_5p = list()
        results_1p_1280h = list()
        lower_bound = 0

        for upper_bound in bins:
            binned_data = df[(df['#samples'] >= lower_bound) & (df['#samples'] < upper_bound)]
            mean_1p, mean_5p, mean_1p_1280h = binned_data[[f'{metric_name}_1p', f'{metric_name}_5p', f'{metric_name}_1p_1280h']].mean().values
            results_1p.append(mean_1p)
            results_5p.append(mean_5p)
            results_1p_1280h.append(mean_1p_1280h)

            lower_bound = upper_bound

        ax = axes.squeeze()[idx]
        ax.plot(results_1p, c='black', label='ProtoPatient')
        ax.plot(results_5p, c='magenta', label='ProtoPatient-5p')
        ax.plot(results_1p_1280h, c='orange', label='ProtoPatient-1p-1280h')

        tick_positions = [i for i in range(len(bins) + 1) if i % int(1 / step) == 0]
        tick_labels = [f"$10^{{{i + 1}}}$" for i in range(len(tick_positions))]

        _ = ax.set_xticks(tick_positions, labels=tick_labels)
        ax.set_title(metric_name)
        ax.legend()
        ax.set_ylim(0, 1)
    axes[1].set_xlabel('#samples')


print('ngroups:', len([10 ** i for i in np.arange(1, 5, 0.08)]))
plot_metrics(full_results, step=0.08)