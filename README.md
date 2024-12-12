
# Boulder Vowels

### [Survey](https://cuboulder.qualtrics.com/jfe/form/SV_8qOvuIxcOhwKpWm)

## Fetching data from Qualtrics
`qualtrics-fetcher.py` downloads wav files to `./corpus/unaligned`

Make sure `.env` is set up with the Qualtrics API key.

## Generating Texgrids
Install the MFA conda environment
```shell
conda config --add channels conda-forge
conda install montreal-forced-aligner
```

Align the corpus
```shell
conda activate aligner
mfa align ./corpus/unaligned ./corpus/dict/dict.txt  english_us_arpa ./corpus/aligned --single_speaker --use_postgres --auto_server
```
After alignment, deactivate the conda environment to use a newer python version for the rest of the project

## Generating the Corpus
`models.py`
```python
from models import Corpus
# reaload=True will load the corpus from the texgrids instead of the pkl
corpus = Corpus(reaload=True)
```
