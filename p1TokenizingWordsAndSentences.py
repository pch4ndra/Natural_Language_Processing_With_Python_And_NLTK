# Natural Language Processing With Python and NLTK p.1 Tokenizing words and Sentences

# import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# tokenizing - word tokenizer... sentence tokenizer = way of splitting strings
# lexicon and corporas
# corpora - body of text. ex: medical journal, presidential speeches, English language
# lexicon - words and their means

# investor-speak... regular english-speak

# investor speak 'bull' = someone who is positive about the market
# english speak 'bull' = scary animal you don't want running @ you

example_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is pinkish-blue. You should not eat cardboard."

##print(sent_tokenize(example_text))
##
##print(word_tokenize(example_text))


for i in word_tokenize(example_text):
    print(i)
