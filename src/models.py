import pickle
from ast import literal_eval
from dataclasses import dataclass, field
from os import listdir
from os.path import splitext
from typing import Callable, Literal

import numpy as np
from numpy import ndarray
from p_tqdm import p_umap
from parselmouth import Formant, PraatError, Sound, Spectrogram, read

np.seterr(all="raise")


class Utils:
    @staticmethod
    def is_vowel(phone_str: str) -> bool:
        return len(phone_str) == 3 and phone_str[0] in ["A", "E", "I", "O", "U"]

    @staticmethod
    def is_rhotic(phone_str: str) -> bool:
        return phone_str[:2] in ["ER", "AX"]

    @staticmethod
    def is_diphthong(phone_str: str) -> bool:
        return phone_str[:2] in ["AW", "AY", "EY", "OW", "OY"]


@dataclass
class Means:
    f1: float = 0.0
    f2: float = 0.0
    f3: float = 0.0


@dataclass
class Word:
    word: str
    phone: str
    start: float
    end: float
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray
    means1: Means = field(default_factory=Means)
    means2: Means = field(default_factory=Means)

    def __post_init__(self):
        length = min(len(self.f1), len(self.f2), len(self.f3))
        assert length > 1, f"Empty slice for f1: {len(self.f1)}, f2: {len(self.f2)}, f3: {len(self.f3)}"
        self.f1 = self.f1[:length]
        self.f2 = self.f2[:length]
        self.f3 = self.f3[:length]


@dataclass
class Monophthong(Word):
    def __post_init__(self):
        super().__post_init__()
        self.means1 = Means(float(np.mean(self.f1)), float(np.mean(self.f2)), float(np.mean(self.f3)))


@dataclass
class Diphthong(Word):
    def __post_init__(self):
        super().__post_init__()

        # remove the first and last 10% of the array
        def split(arr: ndarray) -> tuple[ndarray, ndarray]:
            halfway = len(arr) // 2
            part1 = arr[:halfway]
            part2 = arr[halfway:]

            # try to remove the first and last 10% of the array on both parts
            trimmed1 = part1[int(len(part1) * 0.1) : int(len(part1) * 0.9)]
            trimmed2 = part2[int(len(part2) * 0.1) : int(len(part2) * 0.9)]

            # if the array is still empty, remove the first or last 10% of the array
            if len(trimmed1) < 1:
                trimmed1 = part1[: int(len(part1) * 0.9)]
                if len(trimmed1) < 1:
                    trimmed1 = part1
            if len(trimmed2) < 1:
                trimmed2 = part2[int(len(part2) * 0.1) :]
                if len(trimmed2) < 1:
                    trimmed2 = part2

            return trimmed1, trimmed2

        f1_1, f1_2 = split(self.f1)
        f2_1, f2_2 = split(self.f2)
        f3_1, f3_2 = split(self.f3)

        self.means1 = Means(float(np.mean(f1_1)), float(np.mean(f2_1)), float(np.mean(f3_1)))
        self.means2 = Means(float(np.mean(f1_2)), float(np.mean(f2_2)), float(np.mean(f3_2)))


class Entry:
    # can error
    def __init__(self, id: str, words: list[Word] | None = None):
        self.id = id
        self.words: list[Word] = self.construct_words() if words is None else words
        self.sound: Sound
        self.spectrogram: Spectrogram
        self.formants: Formant

    def init_parselmouth(self):
        self.textgrid = read(f"./corpus/aligned/{self.id}.TextGrid").to_tgt()
        self.sound = Sound(f"./corpus/unaligned/{self.id}.wav")
        self.spectrogram = self.sound.to_spectrogram()
        self.formants = self.sound.to_formant_burg()

    # so it prints nicely
    def __str__(self):
        words = []
        for word in self.words:
            words.append(
                str(
                    {
                        "word": word.word,
                        "phone": word.phone,
                        "start": word.start,
                        "end": word.end,
                        "f1": type(word.f1),
                        "f2": type(word.f2),
                        "f3": type(word.f3),
                    }
                )
            )
        words = [f"\t\t{str(word):.100}..." for word in self.words]
        wordstr = "\n".join(words[:3] + [f"\t\t... {len(words)} total"])
        return f"Entry(\n\tid={self.id},\n\twords=(\n{wordstr}\n\t)\n)"

    # pickle
    def __getstate__(self):
        return {"id": self.id, "words": self.words}

    # unpickle
    def __setstate__(self, state):
        self.id = state["id"]
        self.words = state["words"]

    def filter(self, func: Callable[[Word], bool]) -> "Entry":
        words = list(filter(func, self.words))
        return Entry(self.id, words)

    def construct_words(self) -> list[Word]:
        self.init_parselmouth()

        words = list(self.textgrid.tiers[0])
        vowels = list(filter(lambda intr: Utils.is_vowel(intr.text), self.textgrid.tiers[1]))
        assert len(vowels) == len(words), f"len(vowels) != len(words): {len(vowels)} != {len(words)}"

        # get the formants
        results = []
        for phone, word in zip(vowels, words):
            f1, f2, f3 = [], [], []
            start, end, step = phone.start_time, phone.end_time, self.formants.get_time_step()
            for time in np.arange(start, end, step):
                f1.append(self.formants.get_value_at_time(1, time))
                f2.append(self.formants.get_value_at_time(2, time))
                f3.append(self.formants.get_value_at_time(3, time))

            if len(f1) == 0 or len(f2) == 0 or len(f3) == 0:
                print(f"Skipping word '{word.text}': invalid formant data: {len(f1) = }, {len(f2) = }, {len(f3) = }")
                continue

            word_class = Diphthong if Utils.is_diphthong(phone.text) else Monophthong
            results.append(
                word_class(
                    word.text, phone.text, phone.start_time, phone.end_time, np.array(f1), np.array(f2), np.array(f3)
                )
            )

        return results


class Corpus:
    def __init__(
        self,
        entries: list[Entry] | None = None,
        normalize: Literal["lobanov", "labov_ANAE", "none"] = "none",
        reload: bool = False,
        num_to_load: int = -1,
    ):
        """
        normalize: if True, normalize the corpus
        reload: if True, build the corpus from wav files
        num_to_load: -1 for all, n for n
        """
        self.ids = [name for name, ext in map(splitext, listdir("./corpus/unaligned")) if ext == ".wav"]
        if not entries:
            if reload:
                entries = self.build_corpus(num_to_load)
            else:
                entries = self.load()

        self.entries = entries

        match normalize:
            case "lobanov":
                self.normalize_lobanov()
            case "labov_ANAE":
                self.normalize_labov_ANAE()
            case _:
                pass

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    def __str__(self):
        size = f"Size: {len(self.entries)}"
        unique_words = f"Unique words: {len(set([word.word for entry in self.entries for word in entry.words]))}"
        word_type = f"Word type: {type(self.entries[0].words[0])}"
        formant_type = f"Formant type: {type(self.entries[0].words[0].f1)}"
        return "\n".join([size, unique_words, word_type, formant_type])

    def filter(self, func: Callable[[Entry], bool]) -> "Corpus":
        entries = list(filter(func, self.entries))
        return Corpus(entries=entries)

    def write(self) -> None:
        print("writing corpus")
        with open("./corpus/corpus.pkl", "wb") as f:
            pickle.dump(self.entries, f)

    def load(self) -> list[Entry]:
        print("loading corpus")
        with open("./corpus/corpus.pkl", "rb") as f:
            entries = pickle.load(f)
            assert isinstance(entries, list)
            assert isinstance(entries[0], Entry)
            assert isinstance(entries[0].words[0], Word)
            return entries

    def build_corpus(self, num_to_load: int) -> list[Entry]:
        def build_entry(id: str) -> tuple[str, Entry | Exception]:
            try:
                return id, Entry(id)
            except FileNotFoundError as e:
                return id, e
            except PraatError as e:
                return id, e
            except AssertionError as e:
                return id, e
            except Exception as e:
                print(f"Something went wrong with... Raising error...")
                raise e

        print("loading corpus from raw files")
        num_to_load = num_to_load if num_to_load > 0 else len(self.ids)
        entries: list[Entry] = []
        errors: list[tuple[str, Exception]] = []

        # parallel unordered map that prints progress
        results = p_umap(build_entry, self.ids[:num_to_load], desc="Building corpus")

        for id, res in results:
            if isinstance(res, Exception):
                errors.append((id, res))
            else:
                entries.append(res)

        print(("=" * 10) + "error summary" + ("=" * 10))
        for id, e in errors:
            print(f"{id}: {type(e).__name__}: {e}")
        print("=" * 60)
        print(f"Loaded {len(entries)} entries with {len(errors)} errors")
        print("=" * 60)
        return entries

    def normalize_lobanov(self) -> None:
        """
        Lobanov normalization across all speakers:
        For each formant (F1, F2, F3):
        1. Collect all measurements across all words
        2. Calculate mean and standard deviation
        3. Apply z-score transformation to each measurement
        """
        # Collect all formant values across the corpus
        all_f1 = np.concatenate([word.f1 for entry in self for word in entry.words])
        all_f2 = np.concatenate([word.f2 for entry in self for word in entry.words])
        all_f3 = np.concatenate([word.f3 for entry in self for word in entry.words])

        # Calculate means and standard deviations
        f1_mean, f1_std = np.mean(all_f1), np.std(all_f1)
        f2_mean, f2_std = np.mean(all_f2), np.std(all_f2)
        f3_mean, f3_std = np.mean(all_f3), np.std(all_f3)

        for entry in self:
            normalized_words = []
            for word in entry.words:
                f1_norm = (word.f1 - f1_mean) / f1_std
                f2_norm = (word.f2 - f2_mean) / f2_std
                f3_norm = (word.f3 - f3_mean) / f3_std

                word_class = Diphthong if Utils.is_diphthong(word.phone) else Monophthong
                normalized_words.append(
                    word_class(
                        word=word.word,
                        phone=word.phone,
                        start=word.start,
                        end=word.end,
                        f1=f1_norm,
                        f2=f2_norm,
                        f3=f3_norm,
                    )
                )
            entry.words = normalized_words

    def normalize_labov_ANAE(self) -> None:
        """
        Implements Labov's ANAE normalization method using the Telsur G value.

        This method:
        1. Uses G = 6.896874 (Telsur grand mean from ANAE)
        2. Calculates speaker means (S) from log of F1 and F2
        3. Computes scaling factor F = exp(G - S)
        4. Multiplies original formant values by F
        """
        # Telsur grand mean (G) from ANAE
        G = 6.896874

        for entry in self:
            # Get all F1 and F2 values for this speaker
            all_f1 = np.concatenate([word.f1 for word in entry.words])
            all_f2 = np.concatenate([word.f2 for word in entry.words])

            # Calculate speaker's mean (S) using log of F1 and F2
            S = np.mean(np.log(np.concatenate([all_f1, all_f2])))

            # Calculate scaling factor F
            F = np.exp(G - S)

            # Apply scaling to all formants
            normalized_words = []
            for word in entry.words:
                word_class = Diphthong if Utils.is_diphthong(word.phone) else Monophthong
                normalized_words.append(
                    word_class(
                        word=word.word,
                        phone=word.phone,
                        start=word.start,
                        end=word.end,
                        f1=word.f1 * F,
                        f2=word.f2 * F,
                        f3=word.f3 * F,
                    )
                )
            entry.words = normalized_words
