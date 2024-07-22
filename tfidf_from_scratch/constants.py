"""Module to contain constants related to the project."""

from datetime import datetime

# The required NLTK artefacts
RESOURCES = ["punkt", "stopwords", "wordnet", "omw-1.4"]


RUN_CONFIG = {
    "path": "data/documents.parquet",
    "norm": "l1",
    "smooth_idf": True,
    "add_idf": True,
    "output_path": f"output/vectors_{datetime.now().strftime('%Y%m%d')}.parquet",
}
