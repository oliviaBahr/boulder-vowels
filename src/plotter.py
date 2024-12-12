from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

from models import Corpus, Utils

IPA_DICT = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AX": "ə",
    "AXR": "ɚ",
    "AY": "aɪ",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "IH": "ɪ",
    "IX": "ɨ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
    "UX": "ʉ",
}


def plot_monopthongs(corpus: Corpus, ax):
    # Dictionary to store F1 and F2 values for each vowel
    all_data: dict[str, dict[str, list[float]]] = {}

    for entry in corpus:
        for vowel, words in groupby(entry.words, lambda word: word.phone[:2]):
            if not Utils.is_diphthong(vowel):
                if vowel not in all_data:
                    all_data[vowel] = {"f1": [], "f2": []}
                word_list = list(words)
                f1_means = [word.means1.f1 for word in word_list]
                f2_means = [word.means1.f2 for word in word_list]
                all_data[vowel]["f1"].append(float(np.mean(f1_means)))
                all_data[vowel]["f2"].append(float(np.mean(f2_means)))

    # Plot each vowel with a different color
    for vowel, data in all_data.items():
        if data["f1"] and data["f2"]:  # Only plot if there's data
            ax.scatter(data["f2"], data["f1"], alpha=0.5, label=IPA_DICT.get(vowel, vowel))

    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.grid(True)
    ax.legend()


def plot_diphthongs(corpus: Corpus, ax, diphthongs_to_plot):
    # Dictionary to store F1 and F2 values for each vowel
    all_data: dict[str, dict[str, list[float]]] = {}

    for entry in corpus:
        for word in entry.words:
            vowel = word.phone[:2]  # Take first two characters as vowel identifier
            if Utils.is_diphthong(word.phone) and vowel in diphthongs_to_plot:
                # Skip words with NaN values
                if np.isnan(word.means1.f1) or np.isnan(word.means1.f2) or np.isnan(word.means2.f1) or np.isnan(word.means2.f2):
                    continue

                if vowel not in all_data:
                    all_data[vowel] = {"f1_start": [], "f2_start": [], "f1_end": [], "f2_end": []}

                # Store both measurements
                all_data[vowel]["f1_start"].append(word.means1.f1)
                all_data[vowel]["f2_start"].append(word.means1.f2)
                all_data[vowel]["f1_end"].append(word.means2.f1)
                all_data[vowel]["f2_end"].append(word.means2.f2)

    # Plot each diphthong
    for vowel, data in all_data.items():
        if data["f1_start"] and data["f2_start"]:  # Only plot if there's data
            # Get two colors for the gradient
            start_color = plt.get_cmap("coolwarm")(0.2)  # Cool blue
            end_color = plt.get_cmap("coolwarm")(0.8)  # Warm red

            # Plot points for start and end with respective colors
            ax.scatter(data["f2_start"], data["f1_start"], alpha=0.5, color=start_color, label=f"{IPA_DICT.get(vowel, vowel)} start")
            ax.scatter(data["f2_end"], data["f1_end"], alpha=0.5, color=end_color, label=f"{IPA_DICT.get(vowel, vowel)} end")

            # Draw lines connecting start and end points with transparent gray
            for i in range(len(data["f1_start"])):
                points = np.array([[data["f2_start"][i], data["f1_start"][i]], [data["f2_end"][i], data["f1_end"][i]]]).T
                line = plt.plot(points[0], points[1], alpha=0.1, color="gray")[0]

            # No need for additional legend entry since start/end points are already labeled

    # Customize the plot
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")

    ax.grid(True)
    # Only create legend if there are labeled artists
    if ax.get_legend_handles_labels()[0]:
        ax.legend()


def plot_monopthong_clouds(corpus: Corpus, ax):
    # Dictionary to store F1 and F2 values for each vowel
    all_data: dict[str, dict[str, list[float]]] = {}

    for entry in corpus:
        for vowel, words in groupby(entry.words, lambda word: word.phone[:2]):
            if not Utils.is_diphthong(vowel) and not Utils.is_rhotic(vowel):
                if vowel not in all_data:
                    all_data[vowel] = {"f1": [], "f2": []}
                word_list = list(words)
                f1_means = [word.means1.f1 for word in word_list]
                f2_means = [word.means1.f2 for word in word_list]
                all_data[vowel]["f1"].append(float(np.mean(f1_means)))
                all_data[vowel]["f2"].append(float(np.mean(f2_means)))

    # Get color map
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(all_data)))

    # Plot each vowel with an ellipse
    for (vowel, data), color in zip(all_data.items(), colors):
        if data["f1"] and data["f2"]:  # Only plot if there's data
            f1_array = np.array(data["f1"])
            f2_array = np.array(data["f2"])

            # Calculate mean point
            mean_f1 = float(np.mean(f1_array))
            mean_f2 = float(np.mean(f2_array))

            # Calculate covariance matrix
            cov = np.cov(f2_array, f1_array)

            # Calculate eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eig(cov)

            # Calculate angle of rotation
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

            # Create ellipse
            # Scale factor 1.795 gives us approximately 80% confidence interval
            width, height = 1.795 * np.sqrt(eigenvals)
            ellipse = Ellipse(xy=(mean_f2, mean_f1), width=width, height=height, angle=angle, alpha=0.3, label=IPA_DICT.get(vowel, vowel), color=color)

            ax.add_patch(ellipse)
            # Add text label with larger font size
            ax.text(
                mean_f2,
                mean_f1,
                IPA_DICT.get(vowel, vowel),
                horizontalalignment="center",
                verticalalignment="center",
                color="black",
                fontsize=13,  # Added larger font size
                bbox=dict(facecolor=color, alpha=0.3, edgecolor="none", pad=1),
            )

    # Customize the plot
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")

    ax.grid(True)
    # Only create legend if there are labeled artists
    if ax.get_legend_handles_labels()[0]:
        ax.legend()


def plot_corpus(corpus: Corpus):
    XLIM = (3600, 500)
    YLIM = (1300, 0)
    # Create a figure with a 2x2 grid
    fig = plt.figure(figsize=(10, 10))  # Square figure for equal-sized plots
    gs = GridSpec(2, 2)

    # Monophthongs plot in top-left
    ax_mono = fig.add_subplot(gs[0, 0])
    ax_mono.set_xlim(*XLIM)
    ax_mono.set_ylim(*YLIM)
    ax_mono.set_box_aspect(1)
    ax_mono.set_title("Monophthongs")

    # Plot monophthongs
    plot_monopthong_clouds(corpus, ax_mono)

    # Create three plots for diphthongs in the remaining spaces
    diphthongs = ["AY", "EY", "AW"]
    positions = [(0, 1), (1, 0), (1, 1)]  # Positions for the three diphthong plots

    for diph, pos in zip(diphthongs, positions):
        ax = fig.add_subplot(gs[pos])
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_box_aspect(1)
        ax.set_title(f"{IPA_DICT.get(diph, diph)}")

        # Filter corpus for specific diphthong
        filtered_corpus = map(lambda entry: entry.filter(lambda word: Utils.is_diphthong(word.phone)), corpus)
        plot_diphthongs(Corpus(entries=list(filtered_corpus)), ax, [diph])

    plt.tight_layout()
    plt.show()


def main():
    corpus = Corpus(normalize="labov_ANAE")
    plot_corpus(corpus)


if __name__ == "__main__":
    main()
