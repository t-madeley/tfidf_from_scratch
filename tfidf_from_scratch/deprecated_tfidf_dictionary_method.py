"""Module to contain functions related to calculated TF-IDF calculation. This is no deprecated."""

import math
from typing import Dict, List

# TODO: This method was ultimately rejected, I have retained it for discussion


def compute_tf(
    tokenized_document: List[str],
) -> Dict[str, float]:
    """
    Compute the term frequency (TF) for each term in the given tokenized document.

    Parameters
    ----------
    tokenized_document : List[str]
        A list of preprocessed tokens.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping each term to its normalized term frequency.
    """
    tf = {}
    total_terms = len(tokenized_document)

    # Count the occurrences of each term - We could use defaultdict here instead..
    for token in tokenized_document:
        if token in tf:
            tf[token] += 1
        else:
            tf[token] = 1

    # Normalize the term frequency by document length
    if total_terms > 0:
        for term in tf:
            tf[term] /= total_terms

    # Check we have a TF value for all terms in the document
    assert tf.keys() == set(tokenized_document)

    return tf


def compute_idf(
    tokenized_documents: List[List[str]], smooth: bool = True, add_one: bool = True
) -> Dict[str, float]:
    """
    Compute the inverse document frequency (IDF) for each term in the given documents.

    Parameters
    ----------
    tokenized_documents : List[List[str]]
        A list of lists, where each inner list represents a document and contains preprocessed tokens.
    smooth : bool, optional
        Whether to apply smoothing to the IDF values. If True, adds 1 to the document frequencies and the total number of documents. Default is True.
    add_one : bool, optional
        Whether to add 1 to the IDF values. If True, adds 1 to the computed IDF values to mirror the sklearn behaviour.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping each term to its inverse document frequency.
    """
    idf = {}
    n_documents = len(tokenized_documents)

    # Count the number of documents containing each term - We could use defaultdict here instead..
    for doc in tokenized_documents:
        unique_terms = set(doc)
        for term in unique_terms:
            if term in idf:
                idf[term] += 1
            else:
                idf[term] = 1

    # Compute the IDF for each term
    for term, count in idf.items():
        if smooth:
            idf[term] = math.log((n_documents + 1) / (count + 1))
        else:
            idf[term] = math.log(n_documents / count)
        # TODO: Properly clarify this with sklearn implementation
        if add_one:
            idf[term] += 1

    return idf


def compute_tfidf_for_document(
    tf: Dict[str, int], idf: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute the TF-IDF scores for each term in the given tokenized document using the provided TF and IDF values.

    Parameters
    ----------
    tf : Dict[str, int]
        A dictionary mapping each term to its term frequency.
    idf : Dict[str, float]
        A dictionary mapping each term to its inverse document frequency.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping each term to its TF-IDF score.
    """
    tfidf = {}
    sum_tfidf = 0

    for term, freq in tf.items():
        tfidf[term] = freq * idf[term]
        sum_tfidf += tfidf[term]

    # L1 normalization
    for term in tfidf:
        tfidf[term] /= sum_tfidf

    return tfidf


def tfidf(preprocessed_docs: List[List[str]]) -> List[Dict[str, float]]:
    """
    Compute the TF-IDF scores for each term in all of the given pre-processed documents.

    Parameters
    ----------
    preprocessed_docs : List[List[str]]
        A list of input documents, preprocessed, cleaned, lemmatized and tokenized.

    Returns
    -------
    List[Dict[str, float]]
        A list of dictionaries, where each dictionary represents the TF-IDF scores for a document.
    """
    # Compute IDF for the corpus
    idf = compute_idf(preprocessed_docs)

    # Compute TF-IDF for each document
    tfidf_scores = []
    for doc in preprocessed_docs:
        tf = compute_tf(doc)
        tfidf = compute_tfidf_for_document(tf, idf)
        tfidf_scores.append(tfidf)

    return tfidf_scores
