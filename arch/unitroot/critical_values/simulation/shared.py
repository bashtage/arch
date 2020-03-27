from typing import Any, Dict


def format_dict(d: Dict[Any, Any]):
    return (
        str(d)
        .replace(" ", "")
        .replace("],", "],\n")
        .replace(":", ":\n")
        .replace("},", "},\n")
    )
