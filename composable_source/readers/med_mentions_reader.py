# Copyright 2021 The Forte Authors. All Rights Reserved.
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
"""
The reader that reads MED Mentions dataset into Datapacks.
"""
import os
import logging
from typing import Any, Iterator
from nltk.tokenize.treebank import TreebankWordTokenizer
from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Document, Token
from ftx.medical import MedicalEntityMention

__all__ = ["MedMentionsReader"]


class MedMentionsReader(PackReader):
    """
    MedMentionReader is designed to read in the MedMentions dataset.
    The dataset is from the following paper,
    Sunil Mohan, Donghui Li "MedMentions: A Large Biomedical Corpus
    Annotated with UMLS Concepts."
    arXiv:1902.09476
    Data could be downloaded from
    https://github.com/chanzuckerberg/MedMentions
    Format:
        PMID |t| Title text
        PMID |a| Abstract text
        PMID StartIndex EndIndex MentionTextSegment SemanticTypeID EntityID
        ...
    """

    def _collect(self, med_mentions_directory) -> Iterator[Any]:
        r"""Iterator over med_mentions files in the data_source.
        Args:
            med_mentions_directory: directory to the med_mentions files.
        Returns: Iterator over files in the path with med_mentions extensions.
        """
        logging.info("Reading MedMentions from %s", med_mentions_directory)
        return dataset_path_iterator(med_mentions_directory, "txt")

    def _cache_key_function(self, collection: str) -> str:
        return os.path.basename(collection)

    def _word_tokenizer(self, input_pack: DataPack, text):
        tokenizer = TreebankWordTokenizer()
        for begin, end in tokenizer.span_tokenize(text):
            Token(input_pack, begin, end)

    def _parse_pack(self, collection: str) -> Iterator[DataPack]:
        logging.info("Processing %s.", collection)
        doc = open(collection, "r", encoding="utf8")
        text, pack_name = "", ""
        pack: DataPack = DataPack()
        for line in doc:
            # Each paper or document ends with a blank line
            if not line.strip("\n") and text != "":
                pack.set_text(text, replace_func=self.text_replace_operation)
                Document(pack, 0, len(text))
                self._word_tokenizer(pack, text)
                pack.pack_name = pack_name
                text = ""
                yield pack
            # Fetch the title information, title includes '|t|'
            # and the abstract includes '|a|'.
            elif "|t|" in line or "|a|" in line:
                if text == "":
                    pack = DataPack()
                    pack_name = line.split("|")[0]
                text += "|".join(line.split("|")[2:])
            elif len(line.split("\t")) == 6:
                line_components = line.strip("\n").split("\t")
                start_index = int(line_components[1])
                end_index = int(line_components[2])
                semantic_type_id = line_components[4]
                umls_link = "UMLS:" + line_components[5]
                # Create the entity_mention and save the
                # semantic_type_id and umls_link information.
                entity_mention = MedicalEntityMention(
                    pack, start_index, end_index
                )
                entity_mention.ner_type = semantic_type_id
                entity_mention.umls_link = umls_link
        doc.close()

        if text != "":
            pack.set_text(text, replace_func=self.text_replace_operation)
            Document(pack, 0, len(text))
            pack.pack_name = pack_name
            yield pack
