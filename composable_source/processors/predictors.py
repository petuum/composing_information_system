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
"""This file contains predictors."""

__all__ = [
    "BertPredictor",
]

from forte.processors.base.batch_processor import Predictor


class BertPredictor(Predictor):
    """
    Predictor for sequence tagging Bert models.
    """

    def predict(self, _batch):
        input_ids = _batch["input_tag"]["data"]
        pad_value = self.configs.feature_scheme["input_tag"][
            "extractor"
        ].get_pad_value()
        input_length = (1 - (input_ids == pad_value).int()).sum(dim=1)
        input_ids = input_ids.cuda()
        input_length = input_length.cuda()
        self.model.eval()
        _, preds = self.model(input_ids, input_length, None)
        return {"output_tag": preds}
