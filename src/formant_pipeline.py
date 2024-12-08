import os

import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import textgrid
from montreal_forced_aligner.command_line.align import align_corpus_cli
from montreal_forced_aligner.command_line.g2p import g2p_cli
from parselmouth.praat import call


def is_vowel(phoneme: str) -> bool:
    # CMU vowel phonemes
    vowels = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
    return phoneme.strip("012").upper() in vowels


def make_corpus(words: list[parselmouth.Sound], transcriptions: list[str]):
    transcriptions = open("data/test_words.txt").read().splitlines()
    g2p_cli(transcriptions, "english_us_arpa", "data/test.txt")


def align_corpus(words: list[parselmouth.Sound], transcriptions: list[str]):
    align_corpus_cli(words, transcriptions, "english_us_arpa", "data/test_output")


def get_word_phonemes(word_sound: parselmouth.Sound, transcription: str):
    # Save the word temporarily
    temp_wav = "temp_word.wav"
    temp_txt = "temp_word.txt"
    word_sound.save(temp_wav, "WAV")

    with open(temp_txt, "w") as f:
        f.write(transcription)

    # Run MFA alignment
    align_corpus_cli([temp_wav], [temp_txt], "english_us_arpa", "temp_output")

    # Read the TextGrid output
    tg = textgrid.TextGrid.fromFile("temp_output/temp_word.TextGrid")
    phone_tier = tg[1]  # The phone tier is typically the second tier

    # Extract phonemes and their timings
    phoneme_segments = []
    for interval in phone_tier.intervals:
        if interval.mark:
            # Create a tuple of (phoneme, start_time, end_time, is_vowel)
            phoneme_segments.append((interval.mark, interval.minTime, interval.maxTime, is_vowel(interval.mark)))

    # Clean up temporary files
    os.remove(temp_wav)
    os.remove(temp_txt)
    os.remove("temp_output/temp_word.TextGrid")
    os.rmdir("temp_output")

    return phoneme_segments


def get_spectrogram(sound: parselmouth.Sound):
    spectrogram = sound.to_spectrogram()
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    return X, Y, sg_db


def get_formants(sound: parselmouth.Sound, phoneme_segments=None):
    """Modified to optionally focus only on vowel segments"""
    formants = sound.to_formant_burg()
    all_formants = []
    all_times = []

    # If no segments provided, analyze the whole sound
    if not phoneme_segments:
        time_steps = np.linspace(0, sound.duration, 100)
    else:
        # Only analyze vowel segments
        time_steps = []
        for phoneme, start, end, is_vowel in phoneme_segments:
            if is_vowel:
                # Create more dense sampling for vowels
                time_steps.extend(np.linspace(start, end, 20))

    for formant_num in range(1, 4):
        formant_values = []
        times = []
        for t in time_steps:
            try:
                formant_value = formants.get_value_at_time(formant_num, t)
                if formant_value:
                    formant_values.append(formant_value)
                    times.append(t)
            except:
                continue
        all_formants.append(formant_values)
        all_times.append(times)

    return all_times, all_formants


def plot_words(words: list[parselmouth.Sound], transcriptions: list[str]):
    num_words = len(words)
    _, axes = plt.subplots(1, num_words, figsize=(5 * num_words, 4))
    if num_words == 1:
        axes = [axes]

    for word, transcription, ax in zip(words, transcriptions, axes):
        # Get phoneme segments
        phoneme_segments = get_word_phonemes(word, transcription)

        # Plot spectrogram
        X, Y, sg_db = get_spectrogram(word)
        ax.pcolormesh(X, Y, sg_db, shading="auto")

        # Get and plot formants (only for vowels)
        times, formants = get_formants(word, phoneme_segments)
        for i in range(3):
            ax.plot(times[i], formants[i], "r-", linewidth=1)

        # Annotate phonemes
        for phoneme, start, end, is_vowel in phoneme_segments:
            # Add vertical lines for phoneme boundaries
            ax.axvline(x=start, color="white", linestyle="--", alpha=0.5)
            # Add phoneme labels with different colors for vowels
            color = "yellow" if is_vowel else "white"
            ax.text((start + end) / 2, 4500, phoneme, horizontalalignment="center", color=color, fontweight="bold" if is_vowel else "normal")

        ax.set_title(f"'{transcription}'")
        ax.set_ylim(0, 5000)

    plt.tight_layout()
    plt.show()


def get_words(sound: parselmouth.Sound):
    grid = call(sound, "To TextGrid (silences)", 100, 0, -25, 0.1, 0.1, "silent", "sounding")
    intervals = call(grid, "Get number of intervals", 1)

    words = []
    for i in range(1, intervals + 1):
        label = call(grid, "Get label of interval", 1, i)
        if label == "sounding":
            start = call(grid, "Get start time of interval", 1, i)
            end = call(grid, "Get end time of interval", 1, i)
            words.append(sound.extract_part(start, end))
    return words


def main():
    sound = parselmouth.Sound("data/test.wav")
    words = get_words(sound)
    transcriptions = ["sought", "bite", "kite", "moat", "bought", "led", "lie", "bout", "should"]
    plot_words(words, transcriptions)


if __name__ == "__main__":
    main()
