from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def hist_plot(y, bins, ax_hist, label, color, alpha):
    hist, bin_edges = np.histogram(y, bins=bins)
    width = bin_edges[1] - bin_edges[0]
    x = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]
    y = hist / sum(hist)
    ax_hist.bar(
        x,
        y,
        width=width,
        color=color,
        alpha=alpha,
        label=label,
        edgecolor="black",
    )

def box_plot(y, ax_box, boxprops, medianprops, flierprops):
    ax_box.boxplot(
        y,
        vert=False,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        flierprops=flierprops,
        widths=0.5,
    )
    ax_box.set_yticks([])

def visualize_prob(y_prob, y_pred, thresh, xlabel="model score"):
    fig, (ax_box1, ax_box2, ax_hist) = plt.subplots(
        3,
        figsize=(5, 5),
        height_ratios=[0.1, 0.1, 0.8],
        sharex=True,
        facecolor="lightcyan",
    )

    boxprops = {"facecolor": "limegreen", "color": "green"}
    medianprops = {"color": "black"}
    flierprops = {"marker": "o", "markerfacecolor": "limegreen", "markersize": 3}
    box_plot(y_prob[(1 - y_pred).astype(bool)], ax_box1, boxprops, medianprops, flierprops)
    
    boxprops = {"facecolor": "indianred", "color": "maroon"}
    flierprops = {"marker": "o", "markerfacecolor": "indianred", "markersize": 3}
    box_plot(y_prob[y_pred.astype(bool)], ax_box2, boxprops, medianprops, flierprops)

    bins = 10
    hist_plot(y_prob[(1 - y_pred).astype(bool)], bins, ax_hist, "True Non-SAR", "limegreen", 1.0)
    hist_plot(y_prob[(y_pred).astype(bool)], bins, ax_hist, "True SAR", "indianred", 0.9)
        
    ax_hist.axvline(
        thresh, color="maroon", ymin=0.0, ymax=1.05, linestyle="dashed", linewidth=2
    )
    ax_hist.set_ylabel("proportion")
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_xticks([np.min(y_prob), thresh, np.max(y_prob)])

    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
    
def get_threshold(y, y_prob, recall):
    y_prob_sar = sorted(y_prob[y == 1])
    num_sar_disqualified = int(np.floor(len(y_prob_sar) * (1 - recall)))
    threshold = y_prob_sar[num_sar_disqualified]
    return threshold

def get_results(y, y_prob, recall=None, threshold=None):
    if not threshold:
        threshold = get_threshold(y, y_prob, recall)
    visualize_prob(y_prob, y, threshold, "model score")
    y_pred = [0 if prob < threshold else 1 for prob in y_prob]
    ConfusionMatrixDisplay.from_predictions(y, y_pred, colorbar=False)