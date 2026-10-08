"""How source names are built and grouped: readers create them, stats groups by them."""

HUB_PREFIX = "hf:"


def is_hub_source(source: object) -> bool:
    """Whether a read() argument names a Hugging Face dataset ("hf:<dataset>")."""
    return isinstance(source, str) and source.startswith(HUB_PREFIX)


def hub_source_name(dataset: str, config: str | None, split: str) -> str:
    """The source name of one hub dataset split: "hf:{dataset}/{config}:{split}"."""
    return f"{HUB_PREFIX}{dataset}/{config or 'default'}:{split}"


def source_group(source: str) -> str:
    """The first path component of a source: its name= prefix, first directory, or hf: id.

    "te/news/a.jsonl" -> "te"; "hf:org/dataset/default:train" -> "hf:org/dataset".
    """
    if source.startswith(HUB_PREFIX):
        return source.rsplit("/", 1)[0]  # drop the "/{config}:{split}" that hub_source_name adds
    return source.split("/", 1)[0]
