"""Module to contain tests for the tfidf.py module."""

import unittest
from typing import Dict, List
from unittest import mock

import pytest

from hiscox_tfidf.tfidf import (
    build_term_counts_matrix,
    build_vocabulary,
    calculate_tfidf_vectors,
    compute_idf,
    compute_tf,
    compute_tfidf,
    get_document_counts,
    normalize_tfidf_scores,
)


@pytest.fixture
def sample_tokenized_documents() -> List[List[str]]:
    return [
        ["the", "quick", "brown", "fox"],
        ["jumps", "over", "the", "lazy", "dog"],
        ["the", "dog", "barked", "loudly"],
    ]


@pytest.fixture
def term_to_index() -> Dict[str, int]:
    return {
        "barked": 0,
        "brown": 1,
        "dog": 2,
        "fox": 3,
        "jumps": 4,
        "lazy": 5,
        "loudly": 6,
        "over": 7,
        "quick": 8,
        "the": 9,
    }


def test_build_vocabulary_with_fixture(sample_tokenized_documents):
    vocabulary, n_terms = build_vocabulary(sample_tokenized_documents)
    expected_vocabulary = [
        "barked",
        "brown",
        "dog",
        "fox",
        "jumps",
        "lazy",
        "loudly",
        "over",
        "quick",
        "the",
    ]
    expected_n_terms = 10
    assert vocabulary == expected_vocabulary
    assert n_terms == expected_n_terms


def test_build_vocabulary_empty_corpus():
    with pytest.raises(
        AssertionError,
        match="No terms found in the input tokenized documents, check the input data and pre-processing steps.",
    ):
        build_vocabulary([])


def test_build_term_counts_matrix(sample_tokenized_documents, term_to_index):
    expected_counts = [
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    ]
    term_counts = build_term_counts_matrix(
        sample_tokenized_documents, term_to_index, 10
    )
    assert term_counts == expected_counts


@pytest.mark.parametrize(
    "term_counts, expected_counts",
    [
        ([[0, 1, 1], [1, 2, 0]], [1, 2, 1]),
        ([[1, 0, 0], [0, 0, 1], [0, 1, 0]], [1, 1, 1]),
    ],
)
def test_get_document_counts(term_counts, expected_counts):
    document_counts = get_document_counts(term_counts)
    assert document_counts == expected_counts


@pytest.mark.parametrize(
    "term_counts, norm, expected_tf",
    [
        ([3, 1, 2], "l1", [0.5, 0.1667, 0.3333]),
        ([3, 1, 2], "l2", [0.8018, 0.2673, 0.5345]),
        ([3, 1, 2], None, [3, 1, 2]),
    ],
)
def test_compute_tf(term_counts, norm, expected_tf):
    computed_tf = compute_tf(term_counts, norm)
    # Compare computed_tf with expected_tf allowing for slight numerical differences
    assert len(computed_tf) == len(expected_tf)
    for computed, expected in zip(computed_tf, expected_tf):
        assert round(computed, 4) == expected


@pytest.mark.parametrize(
    "document_counts, n_documents, smooth, add_one, expected_idf",
    [
        ([1, 2, 3], 5, False, False, [1.6094, 0.9163, 0.5108]),
        ([1, 2, 3], 5, True, False, [1.0986, 0.6931, 0.4055]),
        ([1, 2, 3], 5, True, True, [2.0986, 1.6931, 1.4055]),
        ([1, 2, 3], 5, False, True, [2.6094, 1.9163, 1.5108]),
    ],
)
def test_compute_idf(document_counts, n_documents, smooth, add_one, expected_idf):
    """
    Tests the compute_idf function with precomputed values.
    """
    # Call the function with the test parameters
    computed_idf = compute_idf(document_counts, n_documents, smooth, add_one)

    # Compare the computed IDF with the expected IDF
    assert len(computed_idf) == len(expected_idf)
    for computed, expected in zip(computed_idf, expected_idf):
        assert round(computed, 4) == expected


@pytest.mark.parametrize(
    "norm, expected",
    [
        ("l1", [[0.167, 0.333, 0.500], [0.267, 0.333, 0.400]]),
        ("l2", [[0.267, 0.535, 0.802], [0.456, 0.570, 0.684]]),
        (None, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ],
)
def test_normalize_tfidf_scores(norm, expected):
    # Sample TF-IDF matrix
    tfidf_matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    normalized_matrix = normalize_tfidf_scores(tfidf_matrix, norm=norm)
    for expected_row, normalized_row in zip(expected, normalized_matrix):
        for expected, computed in zip(expected_row, normalized_row):
            assert round(computed, 3) == expected


@pytest.mark.parametrize(
    "tf_matrix, idf_vector, expected_tfidf_matrix",
    [
        (
            [[0.5, 0.2], [0.3, 0.6]],  # TF matrix
            [1.0, 2.0],  # IDF vector
            [[0.5, 0.4], [0.3, 1.2]],  # Expected TF-IDF matrix
        ),
        (
            [[0.4, 0.1, 0.3], [0.2, 0.5, 0.4]],
            [2.0, 1.0, 0.5],
            [[0.8, 0.1, 0.15], [0.4, 0.5, 0.2]],
        ),
    ],
)
def test_compute_tfidf(tf_matrix, idf_vector, expected_tfidf_matrix):
    computed = compute_tfidf(tf_matrix, idf_vector)
    assert computed == expected_tfidf_matrix


class TestCalculateTFIDFVectors(unittest.TestCase):
    @mock.patch("hiscox_tfidf.tfidf.normalize_tfidf_scores")
    @mock.patch("hiscox_tfidf.tfidf.compute_tfidf")
    @mock.patch("hiscox_tfidf.tfidf.compute_idf")
    @mock.patch("hiscox_tfidf.tfidf.get_document_counts")
    @mock.patch("hiscox_tfidf.tfidf.compute_tf")
    @mock.patch("hiscox_tfidf.tfidf.build_term_counts_matrix")
    @mock.patch("hiscox_tfidf.tfidf.build_vocabulary")
    def test_calculate_tfidf_vectors(
        self,
        mock_build_vocabulary,
        mock_build_term_counts_matrix,
        mock_compute_tf,
        mock_get_document_counts,
        mock_compute_idf,
        mock_compute_tfidf,
        mock_normalize_tfidf,
    ):
        # Define test input
        tokenized_documents = [
            ["term1", "term2", "term3"],
            ["term2", "term3", "term4"],
        ]
        norm = "l1"
        smooth_idf = False
        add_idf = False

        # Define mocked return values
        mock_build_vocabulary.return_value = (["term1", "term2", "term3", "term4"], 4)
        mock_build_term_counts_matrix.return_value = [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
        ]
        n_docs = len(mock_build_term_counts_matrix.return_value)

        # Call the function
        tfidf_matrix, term_to_index = calculate_tfidf_vectors(
            tokenized_documents, norm=norm, smooth_idf=smooth_idf, add_idf=add_idf
        )

        # Check the execution of subfunctions
        mock_build_vocabulary.assert_called_once_with(tokenized_documents)
        mock_build_term_counts_matrix.assert_called_once_with(
            tokenized_documents, {"term1": 0, "term2": 1, "term3": 2, "term4": 3}, 4
        )
        mock_compute_tf.assert_called()
        mock_get_document_counts.assert_called_once()
        mock_compute_idf.assert_called_once_with(
            mock_get_document_counts.return_value, n_docs, smooth=False, add_one=False
        )
        mock_compute_tfidf.assert_called_once()
        mock_normalize_tfidf.assert_called_once_with(
            mock_compute_tfidf.return_value, norm=norm
        )
