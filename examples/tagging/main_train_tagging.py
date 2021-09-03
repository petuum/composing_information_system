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
import os
import logging
import argparse
import torch
import yaml
from composable_source.trainers.tagging_trainer import TaggingTrainer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir", type=str, help="Directory to the config files."
    )
    args = parser.parse_args()

    config = {
        "config_data": yaml.safe_load(
            open(os.path.join(args.config_dir, "config_data.yml"), "r")
        ),
        "config_model": yaml.safe_load(
            open(os.path.join(args.config_dir, "config_model.yml"), "r")
        ),
        "device": torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    }

    output_path = config["config_data"]["output_path"]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    log_path = os.path.join(output_path, "training.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    trainer: TaggingTrainer = TaggingTrainer(**config)
    trainer.initialize()
    trainer.run()

    # Save training state and model to disk
    output_state_path = os.path.join(output_path, "train_state.pkl")
    output_model_path = os.path.join(output_path, "model.pt")
    trainer.save(output_state_path)
    torch.save(trainer.model, output_model_path)
