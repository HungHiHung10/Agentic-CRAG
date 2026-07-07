"""Backward-compatible ingestion facade.

New code should import from:
- modules.ingestion.loaders
- modules.ingestion.sources
- modules.ingestion.splitters
- modules.retrieval.retriever
- modules.retrieval.vector_store
"""

from modules.ingestion.loaders import *  # noqa: F401,F403
from modules.ingestion.sources import *  # noqa: F401,F403
from modules.ingestion.splitters import *  # noqa: F401,F403
from modules.retrieval.retriever import *  # noqa: F401,F403
from modules.retrieval.vector_store import *  # noqa: F401,F403