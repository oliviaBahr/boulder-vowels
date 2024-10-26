import random
from os.path import join, dirname


if __name__ == "__main__":
    # making it easy on ourselves
    random.seed(0)

    # get word list
    infile = join(dirname(__file__), "unique.txt")
    wlist = open(infile, "r").readlines()
    wlist = [l.strip() + "\n" for l in wlist if l.strip()]

    # double and shuffle
    shuffled = wlist * 2
    random.shuffle(shuffled)

    # lists of 25 words
    outfile = join(dirname(__file__), "distributed.txt")
    with open(outfile, "w") as f:
        for i, line in enumerate(shuffled):
            if i % 25 == 0:
                f.write(f"\n-------------- {(i // 25) + 1} --------------\n")
            f.write(line.split()[1].strip() + "\n")
