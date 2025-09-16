from langchain_community.document_loaders import JSONLoader
from pathlib import Path
from typing import List, Dict

# news files are obtained via GNews (https://pypi.org/project/gnews/)

# use only publisher title and published date as metadata for chroma store, also change structure of the json
def metadata_func(record: dict, metadata: dict) -> dict:
    publisher = record.get("publisher", {})
    metadata["publisher_title"] = publisher.get("title", "")
    metadata["published_date"] = record.get("published date", "")
    return metadata

def load_json_documents(file_path: Path, content_key: str = "description") -> List[Dict]:
    """
    Loads documents from a JSON file using JSONLoader and attaches metadata.

    Args:
        file_path (Path or str): Path to the JSON file.
        content_key (str): The key in each JSON object that contains the main text.

    Returns:
        List[Dict]: List of documents with content and metadata.
    """
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        content_key=content_key,
        metadata_func=metadata_func,
    )
    docs = loader.load()
    return docs
