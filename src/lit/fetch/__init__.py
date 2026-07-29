"""Paper acquisition: metadata lookup and full-text retrieval."""

from .http import HttpClient  # noqa: F401
from .metadata import PaperMeta, resolve_metadata, search_works  # noqa: F401
from .fulltext import FullText, fetch_fulltext  # noqa: F401
