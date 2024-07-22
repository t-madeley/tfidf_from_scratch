"""Module to contain main functions."""

import json
from typing import Dict

import click
from loguru import logger

from hiscox_tfidf.constants import RUN_CONFIG
from hiscox_tfidf.datasets import load_corpus_from_parquet
from hiscox_tfidf.text_pre_processing import perform_full_preprocessing_and_tokenize
from hiscox_tfidf.tfidf import calculate_tfidf_vectors
from hiscox_tfidf.utils import (
    check_and_download_nltk_resources,
    create_tfidf_dataframe_and_save,
)


@click.group()
def cli() -> None:
    """Run the main function of the package.

    To start using it run a command like:

    "hiscox_tfidf calculate_tfidf "

    """
    pass


@cli.command(name="calculate_tfidf")
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=False,
    help="Path to the custom run configuration JSON file, if None then default used.",
)
def calculate_tfidf_bau(config: str):
    """CLI Command to run TF-IDF calculation."""
    if config is None:
        logger.info("Running with the default config from constants.py")
        run_config = RUN_CONFIG
    else:
        with open(config, "r") as file:
            run_config = json.load(file)
            logger.info(f"Loaded config from file successfully.")

    logger.info(f"Commencing TF-IDF calculation using config: {run_config}")

    tfidf_matrix, term_to_index = main_calculate_tfidf(run_config)


def main_calculate_tfidf(run_config: Dict[str, any] = RUN_CONFIG):
    """
    Calculate the TF-IDF matrix and term-to-index mapping for a given corpus using the specified run configuration.

    Parameters
    ----------
    run_config : Dict[str, any], optional
        The run configuration dictionary specifying the parameters for the TF-IDF calculation.
        Default is the `RUN_CONFIG` constant imported from `constants.py`.
        The dictionary should contain the following keys:
        - "path": str
            The path to the parquet file containing the corpus.
        - "norm": str, optional
            The normalization scheme to use for the TF and TF-IDF calculations.
            Possible values are "l1", "l2", or None (default: None).
        - "smooth_idf": bool, optional
            Whether to apply smoothing to the IDF values (default: True).
        - "add_idf": bool, optional
            Whether to add 1 to the IDF values (default: True).
        - "output_path": str, optional
            The path to save the calculated TF-IDF matrix and term-to-index mapping as a DataFrame.
            If not provided, the result will not be saved.

    Returns
    -------
    Tuple[List[List[float]], Dict[str, int]]
        A tuple containing two elements:
        - tfidf_matrix: List[List[float]]
            The calculated TF-IDF matrix represented as a list of lists of floats.
            Each inner list represents a document, and each element in the inner list represents
            the TF-IDF score of a term in that document.
        - term_to_index: Dict[str, int]
            The term-to-index mapping dictionary, where the keys are the unique terms in the corpus
            and the values are their corresponding indices in the TF-IDF matrix.
    """
    logger.info(f"Commencing TF-IDF calculation using config: {run_config}")

    check_and_download_nltk_resources()

    corpus = load_corpus_from_parquet(run_config.get("path"))

    logger.info("Pre-processing corpus")
    preprocessed_corpus = perform_full_preprocessing_and_tokenize(corpus)
    logger.info(f"Corpus pre-processed, found {len(preprocessed_corpus)} documents")

    logger.info("Beginning TF-IDF Calculation..")
    tfidf_matrix, term_to_index = calculate_tfidf_vectors(
        tokenized_documents=preprocessed_corpus,
        norm=run_config.get("norm"),
        smooth_idf=run_config.get("smooth_idf"),
        add_one_sklearn_idf=run_config.get("add_one_sklearn_idf"),
    )

    logger.success("TF-IDF matrix calculated successfully")

    if run_config.get("output_path", None):
        logger.info(
            f"Creating TF-IDF pandas dataframe and saving to {run_config['output_path']}"
        )
        create_tfidf_dataframe_and_save(
            tfidf_matrix, term_to_index, run_config["output_path"]
        )

    return tfidf_matrix, term_to_index
