#  Copyright 2021 The Forte Authors. All Rights Reserved.
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
Training pipeline for building sequence labeling models using pretrained Bert.
"""
import os
import logging
from typing import Iterator, Dict
import pickle
import torch
from texar.torch.data import Batch
from texar.torch.modules import BERTClassifier
from tqdm import tqdm

from forte.pipeline import Pipeline
from forte.data.data_pack import DataPack
from forte.trainer.base.trainer import BaseTrainer
from composable_source.processors.predictors import BertPredictor
from composable_source.utils.utils_trainer import (
    build_lr_decay_scheduler, compute_loss, create_class)


logger = logging.getLogger(__name__)


class TaggingTrainer(BaseTrainer):
    """
    Trainer for tagging
    """
    def __init__(self,
                 config_data: Dict,
                 config_model: Dict,
                 device):
        super().__init__()

        # All the configs
        self.config_data: Dict = config_data
        self.config_model: Dict = config_model
        self.device = device
        self.model = None

    def create_tp_config(self) -> Dict:
        tp_config: Dict = {
            "preprocess": {
                "device": self.device.type
            },
            "dataset": {
                "batch_size": self.config_data["train_batch_size"]
            },
            "request": self.config_model["tp_request"]
        }
        return tp_config

    def create_pack_iterator(self) -> Iterator[DataPack]:
        reader, reader_config = create_class(
            self.config_data["reader"]["class_name"],
            self.config_data["reader"]["config"])
        train_pl: Pipeline = Pipeline()
        train_pl.set_reader(reader, config=reader_config)
        for processor in self.config_model["processors"].values():
            proc, proc_config = create_class(
                processor["class_name"],
                processor["config"])
            train_pl.add(component=proc, config=proc_config)
        train_pl.initialize()
        pack_iterator: Iterator[DataPack] = \
            train_pl.process_dataset(self.config_data["train_path"])

        return pack_iterator

    def build_model(self, num_classes):
        """
        Build model
        :param num_classes:
        :return:
        """
        pretrained_model = self.config_model["pretrained_model_name"]
        self.model = \
            BERTClassifier(
                pretrained_model_name=pretrained_model,
                hparams={
                    "num_classes": num_classes,
                    "clas_strategy": 'time_wise'
                })
        self.model.to(self.device)

    def train_epoch(self, batch_iter, optim, lr_scheduler, pad_value):
        """
        training epoch
        :param batch_iter:
        :param optim:
        :param lr_scheduler:
        :param pad_value:
        :return:
        """
        train_err: int = 0
        train_total: float = 0.0
        train_sentence_len_sum: float = 0.0
        for batch in tqdm(batch_iter):
            optim.zero_grad()
            input_ids = batch["input_tag"]["data"]
            input_length = (1 - (input_ids == pad_value).int()).sum(dim=1)
            output = batch["output_tag"]["data"]
            logits, _ = self.model(input_ids, input_length, None)
            loss = compute_loss(self.model, logits, output)
            loss.backward()
            optim.step()
            lr_scheduler.step()

            batch_train_err = loss.item() * batch.batch_size
            train_err += batch_train_err
            train_total += batch.batch_size
            train_sentence_len_sum += \
                torch.sum(batch["input_tag"]["masks"][0]).item()
        return train_err, train_total, train_sentence_len_sum

    def build_predictor(self, vocab_path):
        """
        Build Predictor
        :param vocab_path:
        :return:
        """
        predictor = BertPredictor()
        predictor_config = self.config_model["predictor"]
        (self.config_model["tp_request"]
                          ["feature_scheme"]
                          ["output_tag"]
                          ["extractor"]
                          ["vocab_path"]) = vocab_path
        predictor_config.update(self.config_model["tp_request"])
        predictor.load(self.model)
        return predictor, predictor_config

    def build_val_pl(self, predictor, evaluator,
                     evaluator_config, predictor_config):
        """
        Build Validation Pipeline
        :param predictor:
        :param evaluator:
        :param evaluator_config:
        :param predictor_config:
        :return:
        """
        val_reader, val_reader_config = create_class(
            self.config_data["reader"]["class_name"],
            self.config_data["reader"]["config"])
        val_pl: Pipeline = Pipeline()
        val_pl.set_reader(val_reader, config=val_reader_config)
        for processor in self.config_model["processors"].values():
            proc, proc_config = create_class(
                processor["class_name"],
                processor["config"])
            val_pl.add(component=proc, config=proc_config)
        val_pl.add(component=predictor, config=predictor_config)
        val_pl.add(component=evaluator, config=evaluator_config)
        return val_pl

    def save_vocab(self, train_preprocessor):
        """
        Save vocabulary for prediction use
        :param train_preprocessor:
        :return:
        """
        output_vocab_path = os.path.join(self.config_data["output_path"],
                                         "vocab.pkl")
        with open(output_vocab_path, "wb") as vocab_file:
            pickle.dump(
                train_preprocessor.
                    request["schemes"]["output_tag"]["extractor"].vocab,
                vocab_file)
        return output_vocab_path

    def train(self):
        """
        Training pipeline.
        """
        train_preprocessor = self.train_preprocessor
        vocab_path = self.save_vocab(train_preprocessor)
        input_extractor = train_preprocessor.\
            request["schemes"]["input_tag"]["extractor"]
        output_extractor = train_preprocessor.\
            request["schemes"]["output_tag"]["extractor"]
        self.build_model(num_classes=len(output_extractor.vocab))
        lr_scheduler, optim = \
            build_lr_decay_scheduler(self.model,
                                     self.config_data["num_train_data"],
                                     self.config_data["train_batch_size"],
                                     self.config_data["num_epochs"],
                                     self.config_data["warmup_proportion"])
        predictor, predictor_config = self.build_predictor(vocab_path)
        evaluator, evaluator_config = create_class(
            self.config_model["evaluator"]["class_name"],
            self.config_model["evaluator"]["config"])
        val_pl = self.build_val_pl(predictor,
                                   evaluator,
                                   evaluator_config,
                                   predictor_config)
        # Start Training
        epoch: int = 0
        logger.info("Start training.")
        while epoch < self.config_data["num_epochs"]:
            self.model.train()
            epoch += 1
            batch_iter: Iterator[Batch] = \
                train_preprocessor.get_train_batch_iterator()
            train_err, train_total, train_sentence_len_sum = \
                self.train_epoch(batch_iter,
                                 optim,
                                 lr_scheduler,
                                 input_extractor.get_pad_value())
            logger.info("%dth Epoch training, "
                        "total number of examples: %d, "
                        "Average sentence length: %0.3f, "
                        "loss: %0.3f",
                        epoch, train_total,
                        train_sentence_len_sum / train_total,
                        train_err / train_total)

            # Build and run validation pipeline
            val_pl.run(self.config_data["val_path"])
            logger.info("%dth Epoch evaluating, "
                        "val result: %s",
                        epoch, evaluator.get_result())
