"""Module to contain functions related to loading the dataset."""

from typing import List

import pandas as pd
from loguru import logger


def load_corpus_from_parquet(
    path: str = "data/documents.parquet", corpus_col: str = "synopsis"
) -> List[str]:
    """Load a corpus from a given parquet file, selecting the relevant corpus column."""
    logger.info(f"Reading parquet file at: {path}")
    df = pd.read_parquet(path, engine="pyarrow")

    assert (
        corpus_col in df.columns
    ), f"Required corpus column not found in dataset, please check your config. Found: {df.columns}"
    logger.info(f"Extracting corpus from column: {corpus_col}")
    corpus = df[corpus_col].tolist()

    return corpus
