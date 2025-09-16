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

def safe_model_dump(obj, _seen=None):
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return None  # safer than injecting invalid strings
    _seen.add(obj_id)

    if isinstance(obj, BaseModel):
        return safe_model_dump(obj.model_dump(mode="python", by_alias=False), _seen)
    elif isinstance(obj, dict):
        return {k: safe_model_dump(v, _seen) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [safe_model_dump(i, _seen) for i in obj if i is not None]
    else:
        return obj


def convert_numpy_types(obj, _seen=None):
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return None
    _seen.add(obj_id)

    if isinstance(obj, dict):
        return {k: convert_numpy_types(v, _seen) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [convert_numpy_types(i, _seen) for i in obj if i is not None]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def get_page_content(doc):
    if hasattr(doc, "page_content"):
        return doc.page_content
    elif isinstance(doc, dict) and "page_content" in doc:
        return doc["page_content"]
    elif hasattr(doc, "get") and callable(doc.get):
        return doc.get("page_content", str(doc))
    else:
        return str(doc)
