import re
from nltk import PorterStemmer

from get_vocablist import get_vocablist


def split(delimiters, string, maxsplit=0):
    pattern = '|'.join(map(re.escape, delimiters))
    return re.split(pattern, string, maxsplit)


def process_email(email_contents):
    """
    Preprocesses a the body of an email and returns a list of word indices.

    Parameters
    ----------
    email_contents : string
        The email content.

    Returns
    -------
    list
        A list of word indices.

    """
    vocab_list = get_vocablist()

    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    words = split(""" @$/#.-:&*+=[]?!(){},'">_<;%\n\r""", email_contents)
    word_indices = []
    stemmer = PorterStemmer()
    for word in words:
        word = re.sub('[^a-zA-Z0-9]', '', word)
        if word == '':
            continue
        word = stemmer.stem(word)
        print word,
        if word in vocab_list:
            idx = vocab_list.index(word)
            word_indices.append(idx)

    return word_indices
