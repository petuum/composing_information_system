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
Example script to show how to use FinancialNewsReader.
"""
import os
import argparse

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.processors.writers import PackIdJsonPackWriter
from composable_source.readers import FinancialNewsReader


def main(dataset_dir: str, output_dir: str):
    """
    Build an NLP pipeline to process Financial News dataset using
    FinancialNewsReader, then write the processed dataset out.
    """
    pipeline = Pipeline[DataPack]()
    pipeline.set_reader(FinancialNewsReader())
    pipeline.add(
        PackIdJsonPackWriter(),
        {
            "output_dir": output_dir,
            "zip_pack": True,
            "indent": 2,
            "overwrite": True,
        },
    )
    pipeline.run(dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="sample_data/financial_news/",
        help="Data directory to read the text files from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Output dir to save the processed datapack.",
    )
    args = parser.parse_args()
    for root, subdirectories, files in os.walk(args.data_dir):
        for subdirectory in subdirectories:
            input_dir = os.path.join(root, subdirectory)
            main(input_dir, args.output_dir)
