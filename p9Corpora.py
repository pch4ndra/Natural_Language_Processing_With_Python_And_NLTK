# NLTK Corpora - Natural Language Processing With Python and NLTK p.9

# To find location of a file
import nltk
print(nltk.__file__)

from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

print(tok[5:15])
