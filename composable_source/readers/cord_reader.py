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
The reader that reads CORD_NERD data into Datapacks.
"""
import os
import logging
import json
from typing import Any, Iterator

from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Document, Title
from onto.cord19research import Abstract, Body
from onto.medical import MedicalEntityMention

__all__ = [
    "CORDNERDReader",
    "CORDReader"
]


class CORDNERDReader(PackReader):
    r""":class:`CORDNERDReader` is designed to read in CORD_NERD dataset.
        https://aistairc.github.io/BENNERD/
    """

    def _collect(self, text_directory: str) -> Iterator[Any]:
        r"""Should be called with param `text_directory` which is a path to a
        folder containing txt files.

        Args:
            text_directory: text directory containing the files.

        Returns: Iterator over paths to .txt files
        """
        logging.info("Reading CORD_NERD from %s", text_directory)
        return dataset_path_iterator(text_directory, self.configs.file_ext)

    # pylint: disable=no-self-use
    def _cache_key_function(self, text_file: str) -> str:
        return os.path.basename(text_file)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        logging.info("Processing %s.", file_path)
        with open(file_path, "r", encoding="utf8") as file:
            text = file.read()

        pack = DataPack()
        pack.set_text(text, replace_func=self.text_replace_operation)
        Document(pack, 0, len(pack.text))
        pack.pack_name = os.path.split(file_path)[-1][:-4]

        ann_path = file_path[:-3] + 'ann'
        if not os.path.exists(ann_path):
            logging.warning('No annotation file for %s', file_path)
            yield pack

        ind = '0'
        with open(ann_path, "r", encoding='utf8') as ann_file:
            for line in ann_file:
                line = line.strip()
                line_components = line.split()
                if len(line_components) < 4 or \
                not line_components[0][1:].isdigit():
                    continue
                if line.startswith('T'):
                    ind = line_components[0]
                    ner_type = line_components[1]
                    span_begin = int(line_components[2])
                    span_end = int(line_components[3])
                    entity = MedicalEntityMention(pack, span_begin, span_end)
                    entity.ner_type = ner_type
                elif line.startswith('N'):
                    if line_components[2] != ind:
                        continue
                    if 'UMLS:' not in line_components[3]:
                        raise Exception(
                            '''The last field should start with `UMLS`.
                            The label we get is: {0}. Lable file path
                            is {1}. '''.format(line, ann_path))
                    link = line_components[3]
                    if 'cui_less' not in link:
                        entity.umls_link = link
        yield pack

    @classmethod
    def default_configs(cls):
        """
        Indicate files with a specific extension to
        be processed
        """
        config = super().default_configs()
        config['file_ext'] = '.txt'
        return config


class CORDReader(PackReader):
    """
    The reader that reads COVID-19 Open Research Dataset Challenge (CORD-19)
    data into Datapacks.
    https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
    """

    def _collect(self, text_directory) -> Iterator[Any]:
        """Should be called with param `text_directory` which is a path to a
        folder containing txt files.

        Args:
            text_directory: text directory containing the files.

        Returns: Iterator over paths to .json files
        """
        logging.info("Reading CORD-19 research data from %s", text_directory)
        return dataset_path_iterator(text_directory, self.configs.file_ext)

    # pylint: disable=no-self-use
    def _cache_key_function(self, text_file: str) -> str:
        return os.path.basename(text_file)

    # pylint: disable=no-self-use
    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        logging.info("Start Processing %s.", file_path)

        pack = DataPack()
        with open(file_path) as file:
            json_text = json.load(file)

            title = json_text["metadata"]["title"]
            abstract = ''
            for entry in json_text['abstract']:
                abstract += entry['text']

            body_text = ""
            for entry in json_text['body_text']:
                body_text += entry['text']

            delimiter = '\n\n'
            text = delimiter.join([title, abstract, body_text])
            pack.set_text(text)

            Document(pack, 0, len(pack.text))
            Title(pack, 0, len(title))
            Abstract(pack, len(title) + len(delimiter),
                     len(title) + len(delimiter) + len(abstract))
            Body(pack,
                 len(title) + 2 * len(delimiter) + len(abstract), len(text))

            pack.pack_name = os.path.splitext(os.path.basename(file_path))[0]
            yield pack

    @classmethod
    def default_configs(cls):
        """
        Indicate files with a specific extension to be processed
        :return: A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - file_ext: define the file extension that the processor
            should process.
        """
        config = super().default_configs()
        config['file_ext'] = '.json'
        return config
