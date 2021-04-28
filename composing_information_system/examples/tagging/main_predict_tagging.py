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
from composable_source.trainers.utils import create_class


class PredictPipeline:
    """
    Predict Pipeline.
    """
    def __init__(self, config):
        predict_path = config["predict_path"]
        self.model = torch.load(os.path.join(predict_path, "model.pt"))
        self.vocab_path = os.path.join(predict_path, "vocab.pkl")
        self.reader = config["reader"]
        self.processors = config["processors"]
        self.evaluator_config = config["evaluator"]
        self.predictor_config = config["predictor"]
        self.predictor_config.update(config["tp_request"])

    def build_predictor(self):
        """
        Build predictor.
        """
        predictor = BertPredictor()
        predictor_config = self.predictor_config
        (predictor_config["feature_scheme"]
                        ["output_tag"]
                        ["extractor"]
                        ["vocab_path"]) = self.vocab_path
        predictor.load(self.model)
        return predictor, predictor_config

    def build_predict_pipeline(self):
        """
        Build a prediction pipeline.
        """
        predictor, predictor_config = self.build_predictor()
        reader, reader_config = create_class(
            self.reader["class_name"],
            self.reader["config"])
        evaluator, evaluator_config = create_class(
            self.evaluator_config["class_name"],
            self.evaluator_config["config"])
        pl: Pipeline = Pipeline()
        pl.set_reader(reader, config=reader_config)
        for processor in self.processors.values():
            proc, proc_config = create_class(
                processor["class_name"],
                processor["config"])
            pl.add(component=proc, config=proc_config)
        pl.add(component=predictor, config=predictor_config)
        pl.add(component=evaluator, config=evaluator_config)
        pl.initialize()
        return pl, evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str,
                    help="Directory to the config files.")
    args = parser.parse_args()

    config_predict = yaml.safe_load(
        open(os.path.join(args.config_dir, "config_predict.yml"), "r"))
    config_model = yaml.safe_load(
        open(os.path.join(args.config_dir, "config_model.yml"), "r"))
    config_predict.update(config_model)

    predict_pl = PredictPipeline(config_predict)
    pipeline, pl_evaluator = predict_pl.build_predict_pipeline()
    entry_type = get_class(config_predict["entry_type"])
    attribute = (config_predict["tp_request"]
                               ["feature_scheme"]
                               ["output_tag"]
                               ["extractor"]
                               ["config"]
                               ["attribute"])
    pack_ind = 0

    for pack in pipeline.process_dataset(config_predict["data_path"]):
        pack_ind += 1
        print("-------------- pack %d ---------------------" % pack_ind)
        for instance in pack.get(Sentence):
            sent = instance.text
            output_tags = []
            for entry in pack.get(entry_type, instance):
                value = getattr(entry, attribute)
                output_tags.append((entry.text, value))
            print("sentence: ", sent)
            print("output_tags: ", output_tags)
            print('\n')
        print(pl_evaluator.get_result())
