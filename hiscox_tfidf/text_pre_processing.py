"""Module to contain text pre-processing functions."""

import re
from typing import Dict, List, Tuple

from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# TODO: Add logs


def basic_text_preprocessing(document: str) -> str:
    """Perform basic text preprocessing including removing extra whitespace, punctuation, numbers, and converting to lowercase.

    Parameters
    ----------
    document : str
        The input text to be preprocessed.

    Returns
    -------
    str
        The preprocessed text.
    """
    # Remove leading and trailing whitespace
    document = document.strip()
    # Convert text to lowercase
    document = document.lower()
    # Remove punctuation
    document = re.sub(r"[^\w\s]", "", document)
    # Remove numbers
    document = re.sub(r"\d+", "", document)
    # Remove punctuation
    document = re.sub(r"[^\w\s]", " ", document)
    # Replace multiple spaces with a single space
    document = re.sub(r"\s+", " ", document)
    # Remove line breaks
    document = document.replace("\n", " ").replace("\r", " ")

    return document


def remove_stopwords(words: List[str]) -> List[str]:
    """
    Remove stop words from a list of words.

    Parameters
    ----------
    words : List[str]
        The tokenized document.

    Returns
    -------
    List[str]
        The tokenized document with stop words removed.
    """
    stop_words = set(stopwords.words("english"))
    return [word for word in words if word not in stop_words]


def lemmatize_words(words: List[str]) -> List[str]:
    """
    Lemmatize a list of words using the nltk WordNetLemmatizer.

    Parameters
    ----------
    words : List[str] The tokenized document.


    Returns
    -------
    List[str] The tokenized document with words lemmatized.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def tokenize_nltk(document: str) -> List[str]:
    """
    Tokenize a document into a list of words using NLTK.

    Parameters
    ----------
    document : str
        The document to tokenize.

    Returns
    -------
    List[str]
        The list of words in the document.
    """
    return word_tokenize(document.lower())


def perform_full_preprocessing_and_tokenize(corpus: List[str]) -> List[List[str]]:
    """Perform all required preprocessing and tokenization steps each document in the corpus.

    This function takes a corpus of documents as input and applies the following preprocessing steps to each document:
    1. Basic text preprocessing: Removes extra whitespace, punctuation, numbers, and converts the text to lowercase.
    2. Tokenization: Tokenizes the preprocessed document into a list of words.
    3. Stop word removal: Removes stop words using NLTK's built-in list of English stop words.
    4. Lemmatization: Lemmatizes each word in the tokenized document using NLTK's WordNetLemmatizer.

    Parameters
    ----------
    corpus : List[str]
        A list of documents, where each document is represented as a string.

    Returns
    -------
    List[List[str]]
        A list of lists, where each inner list represents a preprocessed and tokenized document.
        Each tokenized document is a list of strings, where each string is a lemmatized word.

    Example
    -------
     corpus = [
    ...     "This is a test.",
    ...     "This is a very good test.",
    ... ]
    preprocessed_corpus = perform_full_preprocessing_and_tokenize(corpus)

    returns:   [['test'], ['very', 'good', 'test']]
    """
    cleaned_corpus = []
    for document in corpus:
        pre_proc_document = basic_text_preprocessing(document)
        tokenized_document = tokenize_nltk(pre_proc_document)
        cleaned_document = lemmatize_words(remove_stopwords(tokenized_document))
        cleaned_corpus.append(cleaned_document)

    return cleaned_corpus
