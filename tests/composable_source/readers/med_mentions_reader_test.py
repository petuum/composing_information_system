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
Unit tests for MedMentionsReader.
"""

import unittest
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from ftx.medical import MedicalEntityMention
from composable_source.readers import MedMentionsReader


class MedMentionsReaderTest(unittest.TestCase):
    r"""
    Unittest for MedMentionReader.
    """

    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "sample_data/tests/med_mentions"
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(MedMentionsReader())
        self.nlp.initialize()

    def test_process_next(self):
        doc_exists = False
        expected_text = [
            [
                "DCTN4",
                "chronic Pseudomonas aeruginosa infection",
                "cystic fibrosis",
                "Pseudomonas aeruginosa (Pa) infection",
                "cystic fibrosis",
                "CF",
                "patients",
                "long-term",
                "pulmonary disease",
                "shorter survival",
            ],
            [
                "inhibits",
                "apoptosis",
                "induced",
                "PC12 cells",
                "Nonylphenol",
                "short-chain nonylphenol ethoxylates",
                "NP2 EO",
                "present",
                "aquatic environment",
            ],
        ]
        expected_ner_type = [
            [
                "T116,T123",
                "T047",
                "T047",
                "T047",
                "T047",
                "T047",
                "T101",
                "T079",
                "T047",
                "T169",
            ],
            [
                "T052",
                "T043",
                "T169",
                "T025",
                "T131",
                "T131",
                "T131",
                "T033",
                "T067",
            ],
        ]
        expected_umls_link = [
            [
                "UMLS:C4308010",
                "UMLS:C0854135",
                "UMLS:C0010674",
                "UMLS:C0854135",
                "UMLS:C0010674",
                "UMLS:C0010674",
                "UMLS:C0030705",
                "UMLS:C0443252",
                "UMLS:C0024115",
                "UMLS:C0220921",
            ],
            [
                "UMLS:C3463820",
                "UMLS:C0162638",
                "UMLS:C0205263",
                "UMLS:C0085262",
                "UMLS:C1254354",
                "UMLS:C1254354",
                "UMLS:C1254354",
                "UMLS:C0150312",
                "UMLS:C0563034",
            ],
        ]

        # Get processed pack from the dataset.
        for i, pack in enumerate(self.nlp.process_dataset(self.dataset_path)):
            doc_exists = True
            for j, entity_mention in enumerate(pack.get(MedicalEntityMention)):
                self.assertEqual(expected_text[i][j], entity_mention.text)
                self.assertEqual(
                    expected_ner_type[i][j], entity_mention.ner_type
                )
                self.assertEqual(
                    expected_umls_link[i][j], entity_mention.umls_link
                )

        self.assertTrue(doc_exists)


if __name__ == "__main__":
    unittest.main()
