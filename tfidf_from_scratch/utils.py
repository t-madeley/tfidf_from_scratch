"""Module to contain utility functions."""

from typing import Dict, List

import nltk
import pandas as pd
from loguru import logger

from tfidf_from_scratch.constants import RESOURCES


def check_and_download_nltk_resources():
    """Check and download necessary NLTK resources from constants.RESOURCES."""
    logger.info(f"Checking for NLTK resources: {RESOURCES}")
    for resource in RESOURCES:

        try:
            nltk.data.find(f"{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    logger.info(f"NLTK resources all present or downloaded successfully.")


def format_tfidf_matrix_scores(
    tfidf_matrix: List[List[float]], vocabulary: Dict[str, int]
) -> List[Dict[str, float]]:
    """Format a TF-IDF matrix as a list of dictionaries containing term: TF-IDF score pairs for each document.

    This function allows the output embeddings to be interpreted by a human.

    Parameters
    ----------
    tfidf_matrix : List[List[float]]
        The TF-IDF matrix represented as a list of lists of floats.
    vocabulary : Dict[str, int]
        A dictionary mapping terms to their indices in the TF-IDF matrix.

    Returns
    -------
    List[Dict[str, float]]
        A list of dictionaries, where each dictionary represents a document and contains
        the TF-IDF scores for each term in that document.
    """
    num_docs = len(tfidf_matrix)
    num_terms = len(vocabulary)

    tfidf_scores: List[Dict[str, float]] = []
    for i in range(num_docs):
        doc_scores: Dict[str, float] = {}
        for term, index in vocabulary.items():
            if index < num_terms:
                score = tfidf_matrix[i][index]
                if score != 0:
                    doc_scores[term] = score
        tfidf_scores.append(doc_scores)

    return tfidf_scores


def create_tfidf_dataframe_and_save(
    tfidf_matrix: List[List[float]], term_to_index: Dict[str, int], output_path: str
):
    """
    Create a pandas DataFrame from the TF-IDF matrix and vocabulary, and save it to a file.

    Parameters
    ----------
    tfidf_matrix : List[List[float]]
        The TF-IDF matrix represented as a list of lists.
    term_to_index : Dict[str, int]
        The vocabulary dictionary mapping terms to their indices in the matrix.
    output_path : str
        The path to save the resulting DataFrame as a CSV file.
    """
    # Create a DataFrame from the TF-IDF matrix
    df_tfidf = pd.DataFrame(tfidf_matrix, columns=list(term_to_index.keys()))

    # Save the DataFrame to a CSV file
    df_tfidf.to_parquet(output_path, index=False)

    print(f"TF-IDF DataFrame saved to: {output_path}")


def ln_approximation(x):
    """Approximate the natural logarithm of a number using an iterative method.

    The function calculates an approximation of the natural logarithm (ln)
    of a given positive number `x` using the formula:

    .. math::
        ln(x) = lim[n→∞] n * (x^(1/n) - 1)

    This formula uses a large `n` to approximate the behavior of the
    natural logarithm. As `n` approaches infinity, the expression
    becomes closer to the actual natural logarithm value.

    Parameters
    ----------
    x : float
        The positive number for which the natural logarithm is to be calculated.
        `x` must be greater than 0.

    Returns
    -------
    float
        The approximate natural logarithm of the input `x`.

    Examples
    --------
    >>> ln(2.71828)  # Approximate natural logarithm of Euler's number
    1.0000043273672432

    """
    logger.info("Using ln approximation funciton..")
    n = 1e10
    return n * ((x ** (1 / n)) - 1)
