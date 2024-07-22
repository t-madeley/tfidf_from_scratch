"""Module to contain functions related to calculated TF-IDF calculation."""

import math
from typing import Dict, List, Tuple

from loguru import logger


def build_vocabulary(tokenized_documents: List[List[str]]) -> Tuple[List[str], int]:
    """
    Build the vocabulary from the tokenized documents.

    Parameters
    ----------
    tokenized_documents : List[List[str]]
        A list of tokenized documents, where each document is represented as a list of strings (tokens).

    Returns
    -------
    Tuple[List[str], int]
        A tuple containing the vocabulary (a sorted list of unique terms) and the number of terms in the vocabulary.
    """
    # Generate the vocabulary by extracting unique terms from the tokenized documents
    vocabulary = sorted(set(term for doc in tokenized_documents for term in doc))

    n_terms = len(vocabulary)

    assert (
        n_terms > 0
    ), "No terms found in the input tokenized documents, check the input data and pre-processing steps."

    return vocabulary, n_terms


def build_term_counts_matrix(
    tokenized_documents: List[List[str]], term_to_index: Dict[str, int], n_terms: int
) -> List[List[int]]:
    """
    Build the term counts matrix from the tokenized documents using the provided term-to-index mapping.

    Parameters
    ----------
    tokenized_documents : List[List[str]]
        A list of tokenized documents, where each document is represented as a list of strings (tokens).
    term_to_index : Dict[str, int]
        A dictionary mapping terms to their indices in the vocabulary.
    n_terms: int
        The number of terms in the vocabulary
    Returns
    -------
    List[List[int]]
        The term counts matrix, represented as a list of lists, where each inner list corresponds to a document
        and contains the count of each term in that document.
    """
    # Create a matrix to store the term counts for each document
    term_counts = [[0] * n_terms for _ in range(len(tokenized_documents))]

    # Populate the term counts matrix
    for i, doc in enumerate(tokenized_documents):
        for term in doc:
            if term in term_to_index:
                term_index = term_to_index[term]
                term_counts[i][term_index] += 1

    return term_counts


def get_document_counts(term_counts: List[List[int]]) -> List[int]:
    """Calculate the document counts for each term in the vocabulary.

    - zip(*term_counts) creates a transposition of the term count matrix
    - Using the transposition, we take a boolean mask converting for `count > 0` (does the document contain the word)
    - We then use a sum to calculate the number of documents containing each word.
    If we have 3 terms and 2 documents eg:

            term_count_matrix:
                        [[0, 1, 1,],
                         [1, 2, 0,]]

            transposition:
                         [[0, 1],
                         [1, 2],
                         [1, 0]]
            document_count:
                         [1, 2, 1]

    Parameters
    ----------
        term_counts (List[List[int]]): A list of lists representing the term counts for each document.
                                       Each inner list represents a document, and each element in the
                                       inner list represents the count of a specific term in that document.

    Returns
    -------
        List[int]: A list of document counts, where each element represents the count of documents
                   containing the corresponding term.
    """
    document_counts = [
        sum(count > 0 for count in counts) for counts in zip(*term_counts)
    ]
    return document_counts


def compute_tf(term_counts: List[int], norm: str = "l1") -> List[float]:
    """
    Compute the term frequency (TF) for each term in the given document.

    Parameters
    ----------
    term_counts : List[int]
        A list of integers representing the count of each term in the document.
    norm : str, optional
        The normalization scheme to use for term frequency. Possible values are:
        - 'l1': L1 normalization (default)
        - 'l2': L2 normalization
        - None: No normalization, return raw term counts

    Returns
    -------
    List[float]
        A list of floats representing the normalized term frequencies.
    """
    if norm == "l1":
        tf = [count / sum(term_counts) for count in term_counts]
    elif norm == "l2":
        norm = sum(x**2 for x in term_counts) ** 0.5
        tf = [count / norm for count in term_counts]
    else:
        tf = term_counts

    return tf


def compute_idf(
    document_counts: List[int],
    n_documents: int,
    smooth: bool = True,
    add_one_sklearn_idf: bool = True,
) -> List[float]:
    """
    Compute the inverse document frequency (IDF) for each term in the given documents.

    Parameters
    ----------
    document_counts : List[int]
        A list of integers representing the count of documents containing each term.
    n_documents : int
        The total number of documents in the corpus.
    smooth : bool, optional
        Whether to apply smoothing to the IDF values. If True, adds 1 to the document frequencies and the total number of documents. Default is True.
    add_one_sklearn_idf : bool, optional
        Whether to add 1 to the IDF values. If True, adds 1 to the computed IDF values. Default is True.

    Returns
    -------
    List[float]
        A list of floats representing the inverse document frequencies.
    """
    if smooth:
        idf = [math.log((n_documents + 1) / (count + 1)) for count in document_counts]
    else:
        idf = [math.log(n_documents / count) for count in document_counts]

    if add_one_sklearn_idf:
        idf = [value + 1 for value in idf]

    return idf


def normalize_tfidf_scores(
    tfidf_matrix: List[List[float]], norm: str = "l1"
) -> List[List[float]]:
    """
    Normalize the TF-IDF matrix using the specified normalization scheme.

    Parameters
    ----------
    tfidf_matrix : List[List[float]]
        The TF-IDF matrix represented as a list of lists.
    norm : str, optional
        The normalization scheme to use for TF-IDF scores. Possible values are:
        - 'l1': L1 normalization
        - 'l2': L2 normalization (default)
        - None: No normalization

    Returns
    -------
    List[List[float]]
        The normalized TF-IDF matrix.
    """
    if norm == "l1":
        normalized_matrix = [
            [value / sum(row) for value in row] for row in tfidf_matrix
        ]
    elif norm == "l2":
        normalized_matrix = [
            [value / (sum(x**2 for x in row) ** 0.5) for value in row]
            for row in tfidf_matrix
        ]
    else:
        normalized_matrix = tfidf_matrix

    return normalized_matrix


def compute_tfidf(
    tf_matrix: List[List[float]], idf_vector: List[float]
) -> List[List[float]]:
    """
    Compute the TF-IDF scores given the TF matrix and IDF vector.

    Parameters
    ----------
    tf_matrix : List[List[float]]
        The term frequency (TF) matrix represented as a list of lists.
    idf_vector : List[float]
        The inverse document frequency (IDF) vector represented as a list of floats.

    Returns
    -------
    List[List[float]]
        The TF-IDF matrix represented as a list of lists.
    """
    # Compute TF-IDF scores
    tfidf_matrix = [
        [tf_value * idf_value for tf_value, idf_value in zip(tf_row, idf_vector)]
        for tf_row in tf_matrix
    ]

    return tfidf_matrix


def calculate_tfidf_vectors(
    tokenized_documents: List[List[str]],
    norm: str = "l1",
    smooth_idf: bool = True,
    add_one_sklearn_idf: bool = True,
) -> Tuple[List[List[float]], Dict[str, int]]:
    """Compute the TF-IDF scores for each term in the given tokenized documents and returns a matrix with the scores and the vocabulary.

    Parameters
    ----------
    tokenized_documents : List[str]
        A list of input documents as strings.
    norm : str, optional
        The normalization scheme to use for term frequency AND TF-IDF scores. Possible values are:
        - 'l1': L1 normalization (default)
        - 'l2': L2 normalization
        - None: No normalization, user raw term counts
    smooth_idf : bool, optional
        Whether to apply smoothing to the IDF values. If True, adds 1 to the document frequencies and the total number of documents. Default is True.
    add_one_sklearn_idf : bool, optional
        Whether to add 1 to the IDF values. If True, adds 1 to the computed IDF values from the sklearn implementation. Default is True.

    Returns
    -------
    Tuple[List[List[float]], List[str]]
        A tuple containing:
        - A matrix (list of lists) with shape (n_documents, n_features) representing the TF-IDF scores for each document.
        - The term_to_index vocabulary, which is a dictionary of unique terms in the documents with their index in the matrix.
    """
    logger.info(f"Building vocabulary from tokenised documents..")
    vocabulary, n_terms = build_vocabulary(tokenized_documents)
    logger.info(f"Vocabulary created, {n_terms} found after pre-processing")

    # Create a dictionary to map terms to indices in the vocabulary
    term_to_index = {term: index for index, term in enumerate(vocabulary)}

    logger.info(f"Creating term counts..")
    term_counts = build_term_counts_matrix(tokenized_documents, term_to_index, n_terms)

    logger.info(f"Calculating term frequencies with norm={norm}")
    # Compute normalized term frequencies (TF) by iterating over each document's term count.
    tf = [compute_tf(counts, norm=norm) for counts in term_counts]

    # Compute inverse document frequencies (IDF)
    logger.info(f"Creating document counts.. ")
    document_counts = get_document_counts(term_counts)
    n_documents = len(tokenized_documents)
    logger.info(
        f"Calculating inverse document frequency matrix with smooth={smooth_idf}, add_one={add_one_sklearn_idf}"
    )
    idf = compute_idf(document_counts, n_documents, smooth=smooth_idf, add_one_sklearn_idf=add_one_sklearn_idf)

    # Compute TF-IDF scores
    logger.info("Calculating TF-IDF score matrix")
    tfidf_matrix = compute_tfidf(tf_matrix=tf, idf_vector=idf)

    if norm:
        logger.info(f"Normalising TF-IDF score matrix using norm={norm}")
        tfidf_matrix = normalize_tfidf_scores(tfidf_matrix, norm=norm)

    return tfidf_matrix, term_to_index
