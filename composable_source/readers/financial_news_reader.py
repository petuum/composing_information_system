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
The reader that reads financial news data into Datapacks.
"""
import os
from typing import Any, Iterator

from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Document

__all__ = [
    "FinancialNewsReader",
]


class FinancialNewsReader(PackReader):
    """FinancialNewsReader is designed to read in financial news dataset.
       https://github.com/duynht/financial-news-dataset
    """

    def _collect(self, text_directory) -> Iterator[Any]:
        """Should be called with param `text_directory` which is a path to a
        folder containing txt files.

        Args:
            text_directory: text directory containing the files.

        Returns: Iterator over paths to .txt files
        """
        return dataset_path_iterator(text_directory, self.configs.file_ext)

    # pylint: disable=no-self-use
    def _cache_key_function(self, text_file: str) -> str:
        return os.path.basename(text_file)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack = DataPack()
        with open(file_path, "r", encoding="utf8", errors="ignore") as file:
            text = file.readlines()
            clean_text = []
            for line in text:
                line = line.strip()
                if line and not line.startswith('--'):
                    clean_text.append(line)
            clean_text_str = '\n'.join(clean_text)

        pack.set_text(
            clean_text_str, replace_func=self.text_replace_operation)
        Document(pack, 0, len(pack.text))
        pack.pack_name = file_path.split('/')[-1]
        yield pack

    @classmethod
    def default_configs(cls):
        """
        Indicate files with a specific extension to be processed
        """
        config = super().default_configs()
        config['file_ext'] = '.txt'
        return config
