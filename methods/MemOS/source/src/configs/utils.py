import json


def get_json_file_model_schema(json_path: str) -> str:
    """Retrieve the model schema from a JSON file."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("model_schema", None)
