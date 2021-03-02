import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def make_dataset_boxplot(results):
    fig, ax = plt.subplots(figsize=(4.5,3))

    beat_results = []
    downbeat_results = []
    datasets = results.keys()
    for dataset, result in results.items():
        beat_results.append(result['F-measure']['beat'])
        downbeat_results.append(result['F-measure']['downbeat'])

    beat_bplot = plt.boxplot(beat_results, positions=[1,3,5,7], patch_artist=True, vert=True)
    downbeat_bplot = plt.boxplot(downbeat_results, positions=[2,4,6,8], patch_artist=True, vert=True)
    plt.xticks([1.5, 3.5, 5.5, 7.5], datasets)

    ax.axvline(2.5, linestyle='--', linewidth=1, color='lightgray')
    ax.axvline(4.5, linestyle='--', linewidth=1, color='lightgray')
    ax.axvline(6.5, linestyle='--', linewidth=1, color='lightgray')

    # fill with colors
    #colors = ['#FFB000', '#FE6100', '#DC267F', '#1989BE', '#3e4966']
    #colors = ['#7f7f7f', '#d62728', '#ff7f0e', '#']
    
    #downbeat_colors = [(110/255, 188/255, 200/255), (77/255, 88/255, 201/255), (101/255, 180/255, 138/255), (227/255, 140/255, 60/255)]
    #beat_colors = [(196/255, 236/255, 237/255), (200/255, 203/255, 240/255), (197/255, 233/255, 219/255), (245/255, 219/255, 183/255)]

    downbeat_color = (110/255, 188/255, 200/255)
    beat_color = (196/255, 236/255, 237/255)

    for patch_box, patch_median in zip(beat_bplot['boxes'], beat_bplot['medians']):
        patch_box.set_facecolor(beat_color)
        patch_median.set_color('black')
        patch_median.set_alpha(0.7)

    for patch_box, patch_median in zip(downbeat_bplot['boxes'], downbeat_bplot['medians']):
        patch_box.set_facecolor(downbeat_color)
        patch_median.set_color('black')
        patch_median.set_alpha(0.7)

    #ax.yaxis.set_label_position("right")
    #ax.yaxis.tick_right()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    plt.grid(color='lightgray', axis='y', linestyle='-', linewidth=1)

    custom_lines = [matplotlib.lines.Line2D([0], [0], color=downbeat_color, lw=4),
                    matplotlib.lines.Line2D([0], [0], color=beat_color, lw=4)]

    plt.legend(custom_lines, ['Downbeat', 'Beat'])
    plt.tight_layout()
    plt.savefig('plots/beat_boxplot.pdf')
    plt.savefig('plots/beat_boxplot.png')

if __name__ == '__main__':

    results_file = 'results/test.json'

    with open(results_file, 'r') as fp:
        results = json.load(fp)

    make_dataset_boxplot(results)

    for dataset, result in results.items():
        print(f"{dataset}: avg. F1 beat: {np.mean(results[dataset]['F-measure']['beat']):0.3f}   avg. F1 downbeat: {np.mean(results[dataset]['F-measure']['downbeat']):0.3f}")

