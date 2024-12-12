from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.patheffects import withStroke

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


def get_monophthong_axes(corpus: Corpus, ax: Axes) -> Axes:
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
    return ax


def get_diphthong_axes(corpus: Corpus, ax: Axes, diphthongs_to_plot: list[str]) -> Axes:
    # Dictionary to store F1 and F2 values for each vowel
    all_data: dict[str, dict[str, list[float]]] = {}

    for entry in corpus:
        for word in entry.words:
            vowel = word.phone[:2]  # Take first two characters as vowel identifier
            if Utils.is_diphthong(word.phone) and vowel in diphthongs_to_plot:
                # Skip words with NaN values
                if (
                    np.isnan(word.means1.f1)
                    or np.isnan(word.means1.f2)
                    or np.isnan(word.means2.f1)
                    or np.isnan(word.means2.f2)
                ):
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
            ax.scatter(
                data["f2_start"],
                data["f1_start"],
                alpha=0.5,
                color=start_color,
                label=f"{IPA_DICT.get(vowel, vowel)} start",
            )
            ax.scatter(
                data["f2_end"], data["f1_end"], alpha=0.5, color=end_color, label=f"{IPA_DICT.get(vowel, vowel)} end"
            )

            # Draw lines connecting start and end points with transparent gray
            for i in range(len(data["f1_start"])):
                points = np.array(
                    [[data["f2_start"][i], data["f1_start"][i]], [data["f2_end"][i], data["f1_end"][i]]]
                ).T
                line = plt.plot(points[0], points[1], alpha=0.1, color="gray")[0]

            # No need for additional legend entry since start/end points are already labeled

    # Customize the plot
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")

    ax.grid(True)
    # Only create legend if there are labeled artists
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    return ax


def get_monophthong_cloud_axes(corpus: Corpus, ax: Axes) -> Axes:
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
            ellipse = Ellipse(
                xy=(mean_f2, mean_f1),
                width=width,
                height=height,
                angle=angle,
                alpha=0.3,
                label=IPA_DICT.get(vowel, vowel),
                color=color,
            )

            ax.add_patch(ellipse)
            # Add text label with larger font size
            ax.text(
                mean_f2,
                mean_f1,
                IPA_DICT.get(vowel, vowel),
                horizontalalignment="center",
                verticalalignment="center",
                color="white",
                fontsize=13,
                path_effects=[
                    # Create outline effect with the vowel's color
                    withStroke(linewidth=3, foreground=color)
                ],
            )

    # Customize the plot
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")

    ax.grid(True)
    # Only create legend if there are labeled artists
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    return ax


def get_monophthong_combined_axes(corpus: Corpus, ax: Axes) -> Axes:
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

    # Get color map and shuffle the colors
    base_colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(all_data)))
    # Shuffle colors but ensure reproducibility
    rng = np.random.default_rng(seed=10)  # Set seed for reproducibility
    colors = rng.permutation(base_colors)

    # Define different markers for variety
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "8", "H", "d", "*"]

    # Plot each vowel with both scatter and ellipse
    for (vowel, data), color, marker in zip(all_data.items(), colors, markers):
        if data["f1"] and data["f2"]:  # Only plot if there's data
            f1_array = np.array(data["f1"])
            f2_array = np.array(data["f2"])

            # Plot scatter points with different shapes and transparency
            scatter = ax.scatter(
                f2_array,
                f1_array,
                alpha=0.2,
                color=color,
                marker=marker,
                s=10,
                label=IPA_DICT.get(vowel, vowel),
            )

            # Calculate mean point
            mean_f1 = float(np.mean(f1_array))
            mean_f2 = float(np.mean(f2_array))

            # Calculate and plot ellipse
            cov = np.cov(f2_array, f1_array)
            eigenvals, eigenvecs = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 1.795 * np.sqrt(eigenvals)

            # Add filled ellipse with transparency
            ellipse_fill = Ellipse(
                xy=(mean_f2, mean_f1),
                width=width,
                height=height,
                angle=angle,
                fill=True,
                facecolor=color,
                alpha=0.2,
                edgecolor="none",  # No edge for the filled ellipse
            )
            ax.add_patch(ellipse_fill)

            # Add outline ellipse
            ellipse_outline = Ellipse(
                xy=(mean_f2, mean_f1),
                width=width,
                height=height,
                angle=angle,
                fill=False,
                linewidth=2,
                edgecolor=color,  # Solid outline
            )
            ax.add_patch(ellipse_outline)

            # Add text label
            ax.text(
                mean_f2,
                mean_f1,
                IPA_DICT.get(vowel, vowel),
                horizontalalignment="center",
                verticalalignment="center",
                color="black",
                fontsize=13,
                path_effects=[withStroke(linewidth=3, foreground=color)],
            )

    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.grid(True)
    if ax.get_legend_handles_labels()[0]:
        legend = ax.legend()
        for handle in legend.legend_handles:
            setattr(handle, "_sizes", [100])
            setattr(handle, "alpha", 0.3)
    return ax


def get_monophthong_plot(corpus: Corpus) -> tuple[Figure, Axes]:
    XLIM = (3600, 500)
    YLIM = (1300, 0)

    # Create monophthongs figure
    fig_mono = plt.figure(figsize=(10, 10))
    ax_mono = fig_mono.add_subplot(111)
    ax_mono.set_xlim(*XLIM)
    ax_mono.set_ylim(*YLIM)
    ax_mono.set_box_aspect(1)
    ax_mono.set_title("Monophthongs")
    ax_mono = get_monophthong_combined_axes(corpus, ax_mono)

    return fig_mono, ax_mono


def get_diphthong_plots(corpus: Corpus) -> tuple[Figure, list[Axes]]:
    XLIM = (3600, 500)
    YLIM = (1300, 0)

    # Create diphthongs figure
    fig_diph = plt.figure(figsize=(15, 5))
    diphthongs = ["AY", "EY", "AW"]
    axes_diph = []

    for idx, diph in enumerate(diphthongs):
        ax = fig_diph.add_subplot(1, 3, idx + 1)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_box_aspect(1)
        ax.set_title(f"{IPA_DICT.get(diph, diph)}")

        filtered_corpus = map(lambda entry: entry.filter(lambda word: Utils.is_diphthong(word.phone)), corpus)
        ax = get_diphthong_axes(Corpus(entries=list(filtered_corpus)), ax, [diph])
        axes_diph.append(ax)

    plt.tight_layout()
    return fig_diph, axes_diph


def get_corpus_plots(corpus: Corpus) -> tuple[Figure, Axes, Figure, list[Axes]]:
    fig_mono, ax_mono = get_monophthong_plot(corpus)
    fig_diph, axes_diph = get_diphthong_plots(corpus)
    return fig_mono, ax_mono, fig_diph, axes_diph


def main():
    corpus = Corpus(normalize="labov_ANAE")
    fig_mono, ax_mono, fig_diph, axes_diph = get_corpus_plots(corpus)
    # Save monophthongs plot
    fig_mono.savefig(f"{Path.home()}/Desktop/vowel_plot_mono.png", dpi=300, bbox_inches="tight")
    # Save diphthongs plot
    fig_diph.savefig(f"{Path.home()}/Desktop/vowel_plot_diph.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
