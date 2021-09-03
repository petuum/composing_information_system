# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any, Iterator, List, Optional, NamedTuple, Set, Tuple
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.base_reader import PackReader
from forte.common.exception import ProcessorConfigError
from forte.utils import get_class
from ft.onto.base_ontology import Document, Sentence, Token

__all__ = ["CoNLL03Reader"]


class CoNLL03Reader(PackReader):
    r""":class:`CoNLL03Reader` is designed to read in the CoNLL03 dataset.

    The dataset is from the following paper,
    Sang, Erik F., and Fien De Meulder. "Introduction to the CoNLL-2003
    shared task: Language-independent named entity recognition."
    arXiv preprint cs/0306050 (2003).

    Data could be downloaded from
    https://deepai.org/dataset/conll-2003-english

    Data format:
    Data files contains one line "-DOCSTART- -X- -X- O" to represent the
    start of a document. After that, each line will contain one word and
    an empty line represent the start of a new sentence. Each line contains
    four fields, the word, its part-of-speech tag, its chunk tag and its
    named entity tag.

    Example:
        EU NNP B-NP B-ORG
        rejects VBZ B-VP O
        German JJ B-NP B-MISC
        call NN I-NP O
        to TO B-VP O
        boycott VB I-VP O
        British JJ B-NP B-MISC
        lamb NN I-NP O
        . . O O
    """

    class ParsedFields(NamedTuple):
        word: str
        entity_label: Optional[str] = None

    _DEFAULT_FORMAT = ["word", "entity_label"]
    _REQUIRED_FIELDS = ["word", "entity_label"]

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        if not configs.doc_break_str and configs.num_sent_per_doc <= 0:
            raise ProcessorConfigError(
                """Please specify doc_break_str or
                select a positive integer for num_sent_per_doc"""
            )

        if configs.column_format is None:
            raise ProcessorConfigError(
                "Configuration column_format not provided."
            )

        if configs.entity_mention_class is None:
            raise ProcessorConfigError(
                "Configuration entity_mention_class not provided."
            )

        # pylint: disable=attribute-defined-outside-init
        self.entity_mention = get_class(configs.entity_mention_class)
        column_format = configs.column_format
        # Validate column format.
        seen_fields: Set[str] = set()
        self._column_format: List[Optional[str]] = []
        for _, field in enumerate(column_format):
            if field is None:
                self._column_format.append(None)
                continue
            if field not in self.ParsedFields._fields:
                raise ValueError(f"Unsupported field type: '{field}'")
            if field in seen_fields:
                raise ValueError(f"Duplicate field type: '{field}'")
            seen_fields.add(field)
            self._column_format.append(field)
        # Sanity check: certain fields must be present in format.
        for field in self._REQUIRED_FIELDS:
            if field not in seen_fields:
                raise ValueError(f"'{field}' field is required")

    @classmethod
    def default_configs(cls):
        r"""
        Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                "file_ext": 'tsv'
                "num_sent_per_doc": 10,
                "doc_break_str": None,
                "column_format": [
                    "word",
                    None
                    "entity_label"
                ]
            }

        Here:
        `"file_ext"`: str
            A string indicating the extension of the data file.

        `"num_sent_per_doc"`: int
            An integer indicating the number of sentences to be grouped
            in a document. Set it to a positive number
            when doc_break_str is None.

        `"doc_break_str"`: str
            A string indicating the end of a document.

        `"column_format"`: list
            A list of strings indicating which field each column in a
            line corresponds to. The length of the list should be equal to the
            number of columns in the files to be read. Available field types
            include:

            - ``"word"``
            - ``"entity_label"``

            If a column should be ignored, fill in `None` at the corresponding
            position.

            .. note::
                A `None` field means that column in the dataset file will be
                ignored during parsing.

        `"entity_mention_class"`: str
            Which entity mention class you want to use. For example:
            "ft.onto.base_ontology.EntityMention"

        """
        return {
            "file_ext": ".txt",
            "num_sent_per_doc": -1,
            "doc_break_str": None,
            "column_format": cls._DEFAULT_FORMAT,
            "entity_mention_class": None,
        }

    def _collect(self, conll_directory) -> Iterator[Any]:
        r"""Iterator over conll files in the data_source.

        Args:
            conll_directory: directory to the conll files.

        Returns: Iterator over files in the path with conll extensions.
        """
        logging.info("Reading .conll from %s", conll_directory)
        return dataset_path_iterator(conll_directory, self.configs.file_ext)

    def _cache_key_function(self, collection: str) -> str:
        return os.path.basename(collection)

    def _parse_line(self, line: str) -> "ParsedFields":
        parts = line.split()
        fields = {}
        for field, part in zip(self._column_format, parts):
            if field is not None:
                fields[field] = part
        return self.ParsedFields(**fields)

    def _if_break_doc(self, line: str, num_sent: int) -> bool:
        if self.configs.doc_break_str and self.configs.doc_break_str in line:
            return True
        if (
            line == ""
            and num_sent > 0
            and self.configs.num_sent_per_doc > 0
            and num_sent % self.configs.num_sent_per_doc == 0
        ):
            return True
        return False

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        def finish_pack():
            text = " ".join(words)
            pack.set_text(text, replace_func=self.text_replace_operation)
            _ = Document(pack, 0, len(text))

        start_new_doc: bool = True
        num_sent = 1
        with open(file_path, encoding="utf8") as doc:
            for line in doc:
                if start_new_doc:
                    pack = DataPack()

                    words: List = []
                    offset = 0
                    has_rows = False

                    sentence_begin = 0

                    # auxiliary structures
                    current_entity_mention: Optional[Tuple[int, str]] = None
                    start_new_doc = False

                line = line.strip()
                if line == "":
                    if not has_rows:
                        continue
                    # add sentence
                    Sentence(pack, sentence_begin, offset - 1)
                    sentence_begin = offset
                    has_rows = False
                    num_sent += 1

                if self._if_break_doc(line, num_sent):
                    if words != []:
                        finish_pack()
                        yield pack
                    start_new_doc = True
                    continue

                if line != "" and not line.startswith("#"):
                    fields = self._parse_line(line)

                    assert fields.word is not None
                    word_begin = offset
                    word_end = offset + len(fields.word)

                    # add tokens
                    Token(pack, word_begin, word_end)
                    # add entity mentions
                    current_entity_mention = self._process_entity_annotations(
                        pack,
                        fields.entity_label,
                        word_begin,
                        current_entity_mention,
                    )
                    words.append(fields.word)
                    offset = word_end + 1
                    has_rows = True

        if has_rows:
            num_sent += 1
            Sentence(pack, sentence_begin, offset - 1)
        if words != []:
            finish_pack()
            yield pack

    def _process_entity_annotations(
        self,
        pack: DataPack,
        label: Optional[str],
        word_begin: int,
        current_entity_mention: Optional[Tuple[int, str]],
    ) -> Optional[Tuple[int, str]]:

        if label is None:
            return None

        ner_type = label.split("-")[-1]
        if not current_entity_mention:
            return (word_begin, ner_type)

        if (
            label[0] == "O"
            or label[0] == "B"
            or (label[0] == "I" and ner_type != current_entity_mention[1])
        ):
            # Exiting a span, add and then reset the current span.
            if current_entity_mention[1] != "O":
                entity = self.entity_mention(
                    pack, current_entity_mention[0], word_begin - 1
                )
                entity.ner_type = current_entity_mention[1]
            current_entity_mention = (word_begin, ner_type)

        return current_entity_mention
