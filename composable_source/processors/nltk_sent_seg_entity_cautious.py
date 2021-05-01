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
NLTK sentence tokenizer with cautions of EntityMention boundaries.
"""

__all__ = [
    "NLTKSentSegEntityCautious",
]

from typing import List
from nltk import PunktSentenceTokenizer
from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.data.ontology import Annotation
from forte.utils import get_class
from forte.common.exception import ProcessorConfigError
from ft.onto.base_ontology import Sentence


class NLTKSentSegEntityCautious(PackProcessor):
    r"""A wrapper of NLTK sentence tokenizer. It makes sure
    that the boundaries are not within any EntityMention spans.
    """
    # pylint: disable=attribute-defined-outside-init,unused-argument
    def initialize(self, resources: Resources, configs: Config):
        """initialization"""
        super().initialize(resources, configs)

        self.resources = resources
        self.config = Config(configs, self.default_configs())
        if not self.config.entity_mention_type:
            raise ProcessorConfigError(
                "Please specify an entity mention type!")
        self.entity_mention_type = get_class(self.config.entity_mention_type)
        self.sent_splitter = PunktSentenceTokenizer()

    def _process(self, input_pack: DataPack):
        """
        Process input pack
        :param input_pack:
        :return:
        """
        entity_mentions: List[Annotation] = \
                list(input_pack.get(self.entity_mention_type))
        current_begin = 0
        for _, end in self.sent_splitter.span_tokenize(input_pack.text):
            if not self._is_within_entity(entity_mentions, end):
                Sentence(input_pack, current_begin, end)
                current_begin = end + 1

    @staticmethod
    def _is_within_entity(entity_mentions: List[Annotation],
                          position: int) -> bool:
        """
        Determine if a position is within any entity mention span.
        Args:
            Inputs:
                entity_mentions (List[Annotation]): a list of entity mentions
                position (int): an interger indicating current position.
            Output:
                True: position is within one of the entity mention spans.
                False: position is not within one of the entity mention spans.
        """
        for entity_mention in entity_mentions:
            if entity_mention.begin <= position < entity_mention.end:
                return True
        return False

    @classmethod
    def default_configs(cls):
        """
        This defines a basic config structure for NLTKSentSegEntityCautious.
        Returns:
            dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - entity_mention_type: entity mention's type, default is 'None'
        """
        configs = super().default_configs()
        configs.update({
            "entity_mention_type": None,
        })
        return configs
