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
Unit tests for CORDReader
"""
import unittest

from composable_source.readers import CORDReader
from ft.onto.base_ontology import Document, Title
from onto.cord19research import Abstract, Body
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline


class CORDReaderTest(unittest.TestCase):
    r"""
    Unittest for CORDReader.
    """

    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "sample_data/tests/cord19research/"
        self.pipeline = Pipeline[DataPack]()
        self.pipeline.set_reader(CORDReader())
        self.pipeline.initialize()

    def test_process_next(self):
        """
        Test CORDReader
        """
        expected_title = \
            "Systematic Review and Meta-Analysis Kidney Dis Renal Injury by " \
            "SARS-CoV-2 Infection: A Systematic Review Keywords SARS-CoV-2 " \
            "COVID-19 Angiotensin-converting enzyme 2 Renal injury Mechanism"

        expected_abstract = \
            "Background: SARS-CoV-2 infection can cause renal involvement, " \
            "and severe renal dysfunction is more common among patients with " \
            "chronic comorbid conditions, especially patients with " \
            "chronic kidney disease."

        expected_body = \
            " In this review, we summarize the pathogenesis of renal injury " \
            "deriving from SARS-CoV-2 infection by focusing on its etiology, " \
            "pathology, and clinical manifestations."\
            " Renal injury by SARS-CoV-2 is the result of multiple factors. " \
            "Via highly expressed ACE2 in renal tissue, SARS-CoV-2 infection " \
            "fundamentally initiates a mechanism of renal injury. " \
            "Systemic effects such as host immune clearance and immune " \
            "tolerance disorders, endothelial cell injury, thrombus formation, " \
            "glucose and lipid metabolism disorder, and hypoxia aggravate " \
            "this renal injury."

        expected_text = '\n\n'.join([expected_title, expected_abstract,
                                     expected_body])

        # Process pipeline
        data_pack = self.pipeline.process_one(self.dataset_path)
        self.assertIsInstance(data_pack, DataPack)
        self.assertEqual(data_pack.text, expected_text)

        # Test Document
        doc_entries = list(data_pack.get(Document))
        self.assertTrue(len(doc_entries) == 1)
        article = doc_entries[0]
        self.assertIsInstance(article, Document)
        self.assertEqual(article.text, expected_text)

        # Test Title
        title_entries = list(data_pack.get(Title))
        self.assertTrue(len(title_entries) == 1)
        title = title_entries[0]
        self.assertEqual(title.text, expected_title)

        # Test Abstract
        abstract_entries = list(data_pack.get(Abstract))
        self.assertTrue(len(abstract_entries) == 1)
        abstract = abstract_entries[0]
        self.assertEqual(abstract.text, expected_abstract)

        # Test Body
        body_entries = list(data_pack.get(Body))
        self.assertTrue(len(body_entries) == 1)
        body = body_entries[0]
        self.assertEqual(body.text, expected_body)


if __name__ == '__main__':
    unittest.main()
