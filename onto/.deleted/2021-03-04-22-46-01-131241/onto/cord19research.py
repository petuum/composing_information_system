# ***automatically_generated***
# ***source json:../../../../showcase/onto_spec/cord19research.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology cord19research. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation

__all__ = [
    "Abstract",
    "Body",
]


@dataclass
class Abstract(Annotation):
    """
    A span based annotation `Abstract`, used to represent the abstract part of research paper.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class Body(Annotation):
    """
    A span based annotation `Body`, used to represent body part of research paper.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
