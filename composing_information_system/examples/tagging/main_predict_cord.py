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
"""This file makes prediction using the trained bert model."""
import os
import argparse
import yaml
import torch
from forte.pipeline import Pipeline
from forte.utils import get_class
from ft.onto.base_ontology import Sentence
from composable_source.processors import BertPredictor


class PredictPipeline:
    """
    Predict Pipeline.
    """
    def __init__(self, config):
        ner_predict_path = config["ner_predict"]["predict_path"]
        linking_predict_path = config["linking_predict"]["predict_path"]
        self.ner_model = torch.load(
            os.path.join(ner_predict_path, "model.pt"))
        self.linking_model = torch.load(
            os.path.join(linking_predict_path, "model.pt"))

        self.reader = config["reader"]
        self.processors = config["processors"]
        self.ner_predictor_config = config["ner_predict"]
        self.linking_predictor_config = config["linking_predict"]

    def build_predictor(self, model, predictor_config):
        """
        Build predictor.
        """
        predictor = BertPredictor()
        predictor.load(model)
        predict_path = predictor_config["predict_path"]
        pred_config = predictor_config["config"]
        (pred_config["feature_scheme"]
                         ["output_tag"]
                         ["extractor"]
                         ["vocab_path"]) = \
                         os.path.join(predict_path, "vocab.pkl")
        return predictor, pred_config

    def build_predict_pipeline(self):
        """
        Using the saved train state to build a prediction pipeline.
        """
        ner_predictor, ner_predictor_config = \
                        self.build_predictor(self.ner_model,
                                             self.ner_predictor_config)
        linking_predictor, linking_predictor_config = \
                        self.build_predictor(self.linking_model,
                                             self.linking_predictor_config)
        reader = get_class(self.reader["class_name"])()
        pl: Pipeline = Pipeline()
        pl.set_reader(reader, config=self.reader["config"])
        for processor in self.processors.values():
            pl.add(component=get_class(processor["class_name"])(),
                   config=processor["config"])
        pl.add(component=ner_predictor, config=ner_predictor_config)
        pl.add(component=linking_predictor, config=linking_predictor_config)
        pl.initialize()
        return pl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str,
                    help="Directory to the config files.")
    args = parser.parse_args()

    config_predict = yaml.safe_load(
        open(os.path.join(args.config_dir, "config_predict.yml"), "r"))

    predict_pl = PredictPipeline(config_predict)
    pipeline = predict_pl.build_predict_pipeline()
    pack_ind = 0
    for pack in pipeline.process_dataset(config_predict["data_path"]):
        pack_ind += 1
        entry_type = get_class(config_predict["entry_type"])
        ner_attribute = config_predict["ner_predict"]["attribute"]
        linking_attribute = config_predict["linking_predict"]["attribute"]

        print("----------------- pack %d -------------------" % pack_ind)
        for instance in pack.get(Sentence):
            sent = instance.text
            output_ner_tags = []
            output_linking_tags = []
            for entry in pack.get(entry_type, instance):
                ner_value = getattr(entry, ner_attribute)
                linking_value = getattr(entry, linking_attribute)
                if ner_value:
                    output_ner_tags.append((entry.text, ner_value))
                if linking_value:
                    output_linking_tags.append((entry.text, linking_value))
            print("sentence: ", sent)
            print("output_ner_tags: ", output_ner_tags)
            print("output_linking_tags: ", output_linking_tags)
            print('\n')
