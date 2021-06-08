import nltk
import sys
import os
import math
import operator

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    data_path = os.path.join(directory)
    for path in os.listdir(data_path):

        full_path = os.path.join(directory, path)
        page = open(full_path, "r")
        text = page.read()
        files[path] = text

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)
    result = []

    for word in words:
        word = word.lower()
        if word in nltk.corpus.stopwords.words("english"):
            continue
        else:
            if word.islower():
                result.append(word)
            else:
                continue
    
    return result


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = dict()

    for doc in documents:

        for word in documents[doc]:
            if word in words:
                continue
            else:
                count = 0
                for d in documents:
                    if word in documents[d]:
                        count += 1
                words[word] = math.log(float(len(documents) / count))
    
    return words

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs = dict()
    for f in files:
        sum = 0
        for word in query:
            idf = idfs[word]
            sum += files[f].count(word)*idf
        
        tfidfs[f] = sum
    sort = sorted(tfidfs.keys(), key=lambda x: tfidfs[x], reverse=True)
    sort = list(sort)

    return sort[0:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf = dict()

    for s in sentences:
        
        sum = 0
        total = len(sentences[s])

        for word in query:
            count = sentences[s].count(word)
            if word in sentences[s]:
                sum += idfs[word]
        
        idf[s] = (sum, float(count/total))
    
    sort = sorted(idf.keys(), key=lambda x: idf[x], reverse=True)
    sort = list(sort)
    return sort[0:n]


if __name__ == "__main__":
    main()
