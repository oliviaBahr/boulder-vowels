from models import Corpus


def main():
    corpus = Corpus(reload=True)
    print(corpus)


if __name__ == "__main__":
    main()
