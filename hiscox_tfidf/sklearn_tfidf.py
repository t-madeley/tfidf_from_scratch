"""Module to contain functions related to sklearn implementation of TF-IDF used for comparison."""

from typing import Any, Dict, List, Tuple

import nltk
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from hiscox_tfidf.text_pre_processing import perform_full_preprocessing_and_tokenize


def sklearn_tfidf(
    corpus: List[str], run_config: Dict[str, Any]
) -> Tuple[csr_matrix, Dict[str, int]]:
    """Compute the TF-IDF score matrix for each term in the given documents using scikit-learn.

    Parameters
    ----------
    corpus : List[str]
        A list of input documents as strings.
    run_config : Dict[str, Any]
        A dictionary containing the configuration parameters for the TF-IDF computation.
        Expected keys:
        - 'norm': The normalization scheme to use ('l1', 'l2', or None).
        - 'smooth_idf': Whether to apply smoothing to the IDF values (True or False).

    Returns
    -------
    Tuple[csr_matrix, Dict[str, int]]
        A tuple containing:
        - tfidf_matrix: A sparse matrix of shape (n_documents, n_terms) representing the TF-IDF scores.
        - vocabulary: A dictionary mapping terms to their indices in the TF-IDF matrix.
    """
    # Preprocess the documents
    preprocessed_docs = perform_full_preprocessing_and_tokenize(corpus)
    preprocessed_corpus = [" ".join(doc) for doc in preprocessed_docs]

    # Create a TfidfVectorizer object - No pre-processing being done so we can re-use our previous code
    vectorizer = TfidfVectorizer(
        tokenizer=nltk.word_tokenize,
        stop_words=None,
        preprocessor=None,
        lowercase=False,
        use_idf=True,
        norm=run_config["norm"],
        smooth_idf=run_config["smooth_idf"],
    )

    # Compute the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(preprocessed_corpus)

    # Get the vocabulary (term to index mapping)
    vocabulary = vectorizer.vocabulary_

    return tfidf_matrix, vocabulary
