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
Example script to show how to use CORDReader.
"""
import argparse

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.processors.writers import PackNameJsonPackWriter

from composable_source.readers import CORDReader


def main(dataset_dir: str, output_dir: str):
    """
    Build an NLP pipeline to process CORD_NER dataset using
    CORDReader, then write the processed dataset out.
    """
    pipeline = Pipeline[DataPack]()
    pipeline.set_reader(CORDReader())

    pipeline.add(
        PackNameJsonPackWriter(),
        {
            "output_dir": output_dir,
            "indent": 2,
            "overwrite": True,
            "drop_record": True,
        },
    )
    pipeline.run(dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="sample_data/cord_paper/",
        help="Data directory to read the text files from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Output dir to save the processed datapack.",
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
