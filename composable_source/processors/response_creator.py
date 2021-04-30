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
"""
Create human readable output as response
"""
import logging
from collections import defaultdict
from typing import Tuple, Dict, Set, List, DefaultDict, Any
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Token, Sentence, PredicateLink, Title
from onto.medical import MedicalEntityMention
from composable_source.utils.utils import query_preprocess, get_arg_text

logger = logging.getLogger(__name__)

__all__ = [
    "ResponseCreator",
]

URL_PREFIX = 'https://www.ncbi.nlm.nih.gov/search/all/?term='


class ResponseCreator(PackProcessor):
    r"""
    Given datapacks, ResponseCreator output the results in human readable
    format, containing relation, source sentence, and UMLS concepts
    """
    # pylint: disable=useless-super-delegation
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

    @classmethod
    def default_configs(cls):
        """
        This defines a basic config structure for ResponseCreator.
        :return: A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - query_pack_name: the query datapack's name
        """
        config = super().default_configs()
        config.update({
            'query_pack_name': "query"
        })
        return config

    def _process(self, input_pack: MultiPack):
        """
        Process results and generate output
        :param input_pack: MultiPack
        :return:
        """
        query_pack = input_pack.get_pack(self.configs.query_pack_name)

        _, arg0, arg1, _, verb_lemma, is_answer_arg0 = query_preprocess(
            query_pack)

        ent = arg1 if is_answer_arg0 else arg0
        self._process_results(input_pack, ent, verb_lemma, is_answer_arg0)

    def _process_results(self, input_pack: MultiPack, ent: str, verb_lemma: str,
                         is_answer_arg0=True):
        """
        Output relations given input pack and user's interested entity
        :param input_pack: Datapack
        :param ent: entity in user's query
        :param verb_lemma: verb lemma in user's query
        :param is_answer_arg0: if the answer is arg0 or arg1, bool
        :return:
        """
        ent = ent.lower().strip()

        output_relations: DefaultDict[str, List[Any]] = defaultdict(list)
        output_concepts: \
            DefaultDict[int, Dict[str, Dict[str, Set[str]]]] = defaultdict(dict)
        output_titles: DefaultDict[int, Dict[str, Tuple[str, str]]] \
            = defaultdict(dict)

        for pack_idx, pack in enumerate(input_pack.packs):
            if pack.pack_name == self.configs.query_pack_name:
                continue

            result = self._process_datapack(pack_idx, pack, ent,
                                                   verb_lemma, is_answer_arg0)
            for key, item in result.items():
                output_relations[key] = item[0]
                output_titles[pack_idx][key] = item[1]
                output_concepts[pack_idx][key] = item[2]

        intro_relation = u'\u2022Relation:'
        intro_source = u'\u2022Source Sentence:'
        intro_concepts = u'\u2022UMLS Concepts:'

        relations = list({x[0] for x in output_relations.values()})
        relations.sort(key=lambda x: (x[3], x[4]))

        for item in relations:
            triplet = '\t'.join(item[0:3])
            paper_number = output_relations[triplet][1]
            sentence, paper_title = output_titles[paper_number][triplet]

            line_seperator = '=' * 80
            print(f'{line_seperator}\n{intro_relation}\n'
                  f'{triplet}\n{intro_source}\n'
                  f'{sentence}(From Paper: , {paper_title})\n'
                  f'{intro_concepts}')

            leading = ' - '
            sep = '\n\t'
            for umls_ent, desc in \
                    output_concepts[paper_number][triplet].items():
                info = sep.join(desc)
                print(f'{leading}{umls_ent}{sep}{info}')

    def _process_datapack(self, pack_idx: int, pack: DataPack, ent: str,
                          verb_lemma: str, is_answer_arg0: bool):
        title = pack.get_single(entry_type=Title).text
        result: DefaultDict[str, List[Any]] = defaultdict(list)

        for sentence in pack.get(Sentence):
            sent_text = sentence.text.strip()
            if ent not in sent_text.lower():
                continue

            relations: DefaultDict[str, Dict[str, str]] = defaultdict(dict)
            for link in pack.get(PredicateLink, sentence):
                pred = link.get_parent().text

                for token in \
                        pack.get(entry_type=Token, range_annotation=sentence):
                    if token.text == pred and verb_lemma == token.lemma:
                        argument = link.get_child().text
                        relations[pred][link.arg_type] = argument

            for pred, entity in relations.items():
                triplets: List[Tuple[str, str, str, int, int]] = []
                arg0, arg1 = get_arg_text(entity)

                if not arg0 or not arg1:
                    continue

                # check the logic of triplet to filter answers
                if (ent in arg0.lower() and not is_answer_arg0):
                    triplets.append((arg0, pred, arg1, len(arg0), len(arg1)))
                if (ent in arg1.lower() and is_answer_arg0):
                    triplets.append((arg0, pred, arg1, len(arg1), len(arg0)))

                med_entities = self._get_med_ent(pack, sentence, arg0, arg1)
                result = self._collect_triplet_info(triplets, result, pack_idx,
                                                sent_text, title, med_entities)

        return result

    def _collect_triplet_info(self,
                              triplets: List[Tuple[str, str, str, int, int]],
                              result: DefaultDict[str, List[Any]],
                              pack_idx: int, sent_text: str, title: str,
                              med_entities: List[MedicalEntityMention]):
        if not triplets:
            return result

        for triplet in triplets:
            key = '\t'.join(triplet[0:3])
            result[key].append([triplet, pack_idx])
            result[key].append((sent_text, title))

            entity_dict = defaultdict(set)

            for med_ent in med_entities:
                for umls in med_ent.umls_entities:
                    sent = f"Name: {umls.name}\t" \
                           f"CUI: {umls.cui}\t" \
                           f"Learn more at: {URL_PREFIX}{umls.cui}"
                    entity_dict[med_ent.text.lower()].add(sent)

            result[key].append(entity_dict)
        return result

    def _get_med_ent(self, pack: DataPack, sentence: Sentence,
                     arg0: str, arg1: str):
        entities: List[MedicalEntityMention] = []

        for med_ent in pack.get(MedicalEntityMention, sentence):
            if med_ent.text.lower() not in (arg0.lower() or arg1.lower()):
                continue
            entities.append(med_ent)
        return entities