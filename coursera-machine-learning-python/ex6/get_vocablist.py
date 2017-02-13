def get_vocablist():
    """
    Reads the fixed vocabulary list in vocab.txt and returns a list of the words.

    Returns
    -------
    list
        The vocabulary list.
    """
    vocabulary = []
    with open('vocab.txt') as f:
        for line in f:
            idx, word = line.split('\t')
            vocabulary.append(word.strip())
    return vocabulary
