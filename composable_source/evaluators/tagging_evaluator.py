#  Copyright 2020 The Forte Authors. All Rights Reserved.
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
# pylint: disable-msg=too-many-locals
"""Evaluator for Conll03 NER tag."""
import os
from typing import Dict

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.evaluation.base import Evaluator
from forte.utils import get_class
from composable_source.utils.utils_eval import count_exact_match


class BertEvaluatorEntry(Evaluator):
    """
    Evaluator for Bert models. Evaluation strategy is exact match of entries.
    """
    def __init__(self):
        super().__init__()
        self.output_file = "tmp_eval.txt"
        self.scores: Dict[str, float] = {}
        if os.path.isfile(self.output_file):
            os.remove(self.output_file)

    def initialize(self, resources: Resources, configs: Config):
        # pylint: disable=attribute-defined-outside-init,unused-argument
        r"""Initialize the evaluator with `resources` and `configs`.
        This method is called by the pipeline during the initialization.
        Args:
            resources (Resources): An object of class
                :class:`forte.common.Resources` that holds references to
                objects that can be shared throughout the pipeline.
            configs (Config): A configuration to initialize the
                evaluator. This evaluator is expected to hold the
                following (key, value) pairs
                - `"entry_type"` (str): The entry to be evaluated.
                - `"tagging_unit"` (str): The tagging unit that the evalution
                                is preformned on.
                                e.g. "ft.onto.base_ontology.Sentence"
                - `"attribute"` (str): The attribute of the entry to be
                                evaluated.
        """
        super().initialize(resources, configs)
        self.entry_type = get_class(configs.entry_type)
        self.tagging_unit = get_class(configs.tagging_unit)
        self.attribute = configs.attribute

    @classmethod
    def default_configs(cls):
        """This defines a basic config structure for BertEvaluatorEntry
        Returns:
            A dictionary with the default config for this processor.
            entry_type: entry's type, default is None.
            tagging_unit: the unit for tagging task, default is None
            attribute: default is ""
        """
        config = super().default_configs()
        config.update({
            'entry_type': None,
            'tagging_unit': None,
            'attribute': ""
        })
        return config

    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        with open(self.output_file, "a+") as output_file:
            refer_tags = list(ref_pack.get(self.entry_type))
            pred_tags = list(pred_pack.get(self.entry_type))
            refer_count = len(refer_tags)
            pred_count = len(pred_tags)
            exact_match = count_exact_match(refer_tags,
                                            pred_tags,
                                            self.attribute)

            output_file.write(
                    "%d %d %d\n" % (exact_match, refer_count, pred_count)
            )

    def get_result(self) -> Dict:
        exact_match = 0.0
        pred_count = 0.0
        refer_count = 0.0
        with open(self.output_file, "r") as fin:
            for line in fin:
                components = line.split()
                exact_match += float(components[0])
                refer_count += float(components[1])
                pred_count += float(components[2])

        precision = exact_match / refer_count
        recall = exact_match / pred_count
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        self.scores = {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100
        }
        return self.scores
