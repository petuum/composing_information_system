# ***automatically_generated***
# ***source json:../../../../../../../../data/NLP/CASL/composable-showcase/wiki.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology wiki. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from ft.onto.base_ontology import EntityMention
from typing import Optional

__all__ = [
    "WikiEntityMention",
]


@dataclass
class WikiEntityMention(EntityMention):
    """
    A span based annotation class WikiEntityMention, used to represent an Entity Mention with wiki link
    Attributes:
        yago2_entity (Optional[str])
        wiki_url (Optional[str])
        wiki_id (Optional[int])
    """

    yago2_entity: Optional[str]
    wiki_url: Optional[str]
    wiki_id: Optional[int]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.yago2_entity: Optional[str] = None
        self.wiki_url: Optional[str] = None
        self.wiki_id: Optional[int] = None
