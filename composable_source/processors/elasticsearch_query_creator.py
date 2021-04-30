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

# pylint: disable=attribute-defined-outside-init
"""
Query Creator to do NLP analysis for user input and generate query
for ElasticSearch
"""
from typing import Any, Dict, Tuple

from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base import QueryProcessor
from composable_source.utils.utils import query_preprocess

__all__ = [
    "ElasticSearchQueryCreator"
]


class ElasticSearchQueryCreator(QueryProcessor):
    r"""This processor creates a Elasticsearch query and adds it as Query entry
    in the data pack. This query will later used by a Search processor to
    retrieve documents."""

    # pylint: disable=useless-super-delegation
    def __init__(self) -> None:
        super().__init__()

    def _build_query_nlp(self, input_pack: DataPack) -> Dict[str, Any]:
        """Constructs Elasticsearch query that will be consumed by
        Elasticsearch processor with nlp analysis.
        Args:
             input_pack: DataPack
        """
        size = self.configs.size
        field = self.configs.field

        query, arg0, arg1, verb, _, is_answer_arg0 = \
            query_preprocess(input_pack)

        if not arg0 or not arg1:
            processed_query = query

        if is_answer_arg0 is None:
            processed_query = f'{arg0} {verb} {arg1}'.lower()
        elif is_answer_arg0:
            processed_query = f'{arg1} {verb}'.lower()
        else:
            processed_query = f'{arg0} {verb}'.lower()

        return {
            "query": {
                "match_phrase": {
                    field: {
                        "query": processed_query,
                        "slop": 10  # how far we allow the terms to be
                    }
                }
            },
            "size": size
        }

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            "size": 1000,
            "field": "content",
            "query_pack_name": "query"
        })
        return config

    def _process_query(self, input_pack: MultiPack) -> \
            Tuple[DataPack, Dict[str, Any]]:
        """
        process query datapack and return query
        :param input_pack:
        :return:
        """
        query_pack = input_pack.get_pack(self.configs.query_pack_name)
        query_pack.pack_name = self.configs.query_pack_name
        query = self._build_query_nlp(query_pack)
        return query_pack, query
