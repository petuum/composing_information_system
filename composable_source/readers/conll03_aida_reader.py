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
from typing import (Any, Iterator, List, Optional, NamedTuple,
                    Set, Tuple, TextIO, Dict)
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.base_reader import PackReader
from forte.common.exception import ProcessorConfigError
from ft.onto.base_ontology import Document, Sentence, Token
from onto.wiki import WikiEntityMention

__all__ = [
    "CoNLL03LinkReader"
]


# pylint: disable=line-too-long
class CoNLL03LinkReader(PackReader):
    r""":class:`CoNLL03LinkReader` is designed to read in the CoNLL03 dataset
    along with the CoNLL03-AIDA entity annotation.

    The CoNLL03 dataset is from the following paper,
    Sang, Erik F., and Fien De Meulder. "Introduction to the CoNLL-2003
    shared task: Language-independent named entity recognition."
    arXiv preprint cs/0306050 (2003).

    Data could be downloaded from
    https://deepai.org/dataset/conll-2003-english

    The CoNLL03-AIDA annotation can be downloaded from
    https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads
    To use this script, you need to manually separate the annotation files
    into train, dev, and test according to the README downloaded from the
    above link, and put them into the folders of conll03's train/dev/test
    individually.

    Data format:
    Data file contains one line "-DOCSTART- -X- -X- O" to represent the
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

    Annotation format:
    Annotation file also contains one line "-DOCSTART- (<docid>)" to
    represent the start of a document. After that, each line contains
    two or five fields. If the second field is "--NME--", it denotes
    that there is no matching entity for this particular mention.
    Otherwise, each of the column represents:
    1. token index
    2. the corresponding YAGO2 entity
    3. the corresponding Wikipedia URL of the entity
    4. the corresponding Wikipedia ID of the entity
    5. the corresponding Freebase mid


    Example:
        -DOCSTART- (1 EU)
        0	--NME--
        2	Germany	http://en.wikipedia.org/wiki/Germany	11867	/m/0345h
        6	United_Kingdom	http://en.wikipedia.org/wiki/United_Kingdom	31717	/m/07ssc
        9	--NME--
        11	Brussels	http://en.wikipedia.org/wiki/Brussels	3708	/m/0177z
        14	European_Commission	http://en.wikipedia.org/wiki/European_Commission	9974	/m/02q9k
        22	Germany	http://en.wikipedia.org/wiki/Germany	11867	/m/0345h
        28	United_Kingdom	http://en.wikipedia.org/wiki/United_Kingdom	31717	/m/07ssc
    """

    class ParsedFields(NamedTuple):
        word: str
        entity_label: Optional[str] = None

    _DEFAULT_FORMAT = ["word", "entity_label"]
    _REQUIRED_FIELDS = ["word", "entity_label"]

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        if configs.column_format is None:
            raise ProcessorConfigError(
                "Configuration column_format not provided.")
        column_format = configs.column_format
        # Validate column format.
        seen_fields: Set[str] = set()
        # pylint: disable=attribute-defined-outside-init
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
        """
        return {
            "file_ext": '.txt',
            "doc_break_str": '-DOCSTART-',
            "column_format": cls._DEFAULT_FORMAT
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

    def _parse_line(self, line: str) -> ParsedFields:
        parts = line.split()
        fields = {}
        for field, part in zip(self._column_format, parts):
            if field is not None:
                fields[field] = part
        return self.ParsedFields(**fields)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        def finish_pack():
            text = " ".join(words)
            pack.set_text(text,
                        replace_func=self.text_replace_operation)
            _ = Document(pack, 0, len(text))

        linking_file = open(os.path.splitext(file_path)[0] + '.link.tsv', 'r')
        entity_linkings: Dict[int, str] = {}
        start_new_doc: bool = True
        with open(file_path, encoding="utf8") as doc:
            for line in doc:
                if start_new_doc:
                    pack = DataPack()

                    words: List = []
                    offset = 0
                    word_begin = 0
                    has_rows = False
                    sentence_begin = 0
                    token_idx = 0
                    fields = self.ParsedFields("")

                    # auxiliary structures
                    current_entity_mention: \
                        Optional[Tuple[int, int, str]] = None
                    entity_linkings = self._get_entity_linking(linking_file)
                    start_new_doc = False

                line = line.strip()
                if self.configs.doc_break_str in line:
                    if words != []:
                        _ = self._process_entity_annotations(
                            pack, fields.entity_label, word_begin,
                            current_entity_mention, entity_linkings, token_idx,
                            is_last_token=True)
                        finish_pack()
                        yield pack
                        start_new_doc = True
                        continue
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
                        pack, fields.entity_label, word_begin,
                        current_entity_mention, entity_linkings, token_idx
                    )
                    token_idx += 1
                    words.append(fields.word)
                    offset = word_end + 1
                    has_rows = True

                if line == '':
                    if not has_rows:
                        continue
                    # add sentence
                    Sentence(pack, sentence_begin, offset - 1)
                    sentence_begin = offset
                    has_rows = False

        if has_rows:
            Sentence(pack, sentence_begin, offset - 1)
        if words != []:
            _ = self._process_entity_annotations(
                pack, fields.entity_label, word_begin,
                current_entity_mention, entity_linkings, token_idx,
                is_last_token=True)
            finish_pack()
            yield pack
        linking_file.close()

    def _process_entity_annotations(
            self,
            pack: DataPack,
            label: Optional[str],
            word_begin: int,
            current_entity_mention: Optional[Tuple[int, int, str]],
            entity_linkings: Dict[int, str],
            token_idx: int,
            is_last_token: bool = False
    ) -> Optional[Tuple[int, int, str]]:

        if label is None:
            return None

        ner_type = label.split('-')[-1]
        if not current_entity_mention:
            current_entity_mention = (word_begin, token_idx, ner_type)
            return current_entity_mention

        if label[0] == 'O' or label[0] == 'B' or \
            (label[0] == 'I' and ner_type != current_entity_mention[2]) or \
            is_last_token:
            # Exiting a span, add and then reset the current span.
            if current_entity_mention[2] != 'O':
                entity = WikiEntityMention(
                    pack, current_entity_mention[0], word_begin)
                entity.ner_type = current_entity_mention[2]
                begin_token_idx = current_entity_mention[1]
                if begin_token_idx in entity_linkings:
                    line_components = entity_linkings[begin_token_idx].split()
                    if len(line_components) >= 4:
                        entity.yago2_entity = line_components[1]
                        entity.wiki_url = line_components[2]
                        entity.wiki_id = int(line_components[3])
                    del entity_linkings[begin_token_idx]

            current_entity_mention = (word_begin, token_idx, ner_type)

        return current_entity_mention

    def _get_entity_linking(self, linking_file: TextIO) -> Dict[int, str]:
        """
        Read entity list with linking info for a whole document.
        """
        token_idx_to_entity_linking: Dict[int, str] = {}
        for line in linking_file:
            if self.configs.doc_break_str in line \
                and token_idx_to_entity_linking:
                return token_idx_to_entity_linking
            line = line.strip()
            if line != "" and self.configs.doc_break_str not in line:
                token_idx = int(line.split()[0])
                token_idx_to_entity_linking[token_idx] = line
        return token_idx_to_entity_linking
