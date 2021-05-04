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
Unit tests for CORDNERDReader.
"""
import unittest

from composable_source.readers import CORDNERDReader
from onto.medical import MedicalEntityMention
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline


class CORDNERDReaderPipelineTest(unittest.TestCase):
    r"""
    Unittest for CORDNERDReader.
    """

    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "sample_data/tests/CORD_NERD/"
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(CORDNERDReader())
        self.nlp.initialize()

    def test_process_next(self):
        """
        Test if the output of CORDReader is as expected.
        """
        doc_exists = False

        filename = ['24500', '24501']
        expected_text = [
            ['Coronavirus', 'Severe acute respiratory syndrome', 'SARS',
             'coronavirus', 'enzyme', 'host cell', 'virus assembly', 'first',
             'manipulation', 'infection', 'virion', 'early stages',
             'infection', 'Mass spectrometry', 'kinase', 'nearly 200',
             'virion', 'nonstructural protein 3', 'nsp3', 'virion',
             'mass spectrometry', 'Coronaviridae', 'domain architecture',
             'nsp3', 'protein domains', 'Escherichia coli', 'two', 'nsp3',
             'One', 'SARS', 'nucleic acid chaperone-like domain',
             'papain-like proteinase', 'cysteine', 'binding domain',
             'interdomain', 'nsp3', 'virion'],
            ['Bax', 'Ku70', 'DNA', 'end-joining', 'Ku70', 'one', 'apoptosis',
             'Bax', 'mitochondria', 'Ku70', 'Bax', 'fully understood',
             'Ku70', 'Bax', 'growth conditions', 'Bax', 'ubiquitylation',
             'apoptosis', 'wild-type cells', 'Bax', 'Ku70', ') cells',
             'Bax', 'recombinant Ku70', 'extract', 'Ku70', ') cells',
             'Bax', 'proteasome', 'inhibitors', 'Ku70',
             'polyubiquitin chains', 'monoubiquitin', 'Ku70', 'apoptosis',
             'Bax', 'mitochondria', 'Bax', 'proteasome', 'inhibitors',
             'tumor']
        ]
        expected_type = [
            ['CORONAVIRUS', 'DISEASE_OR_SYNDROME', 'CORONAVIRUS',
             'CORONAVIRUS', 'CHEMICAL', 'CHEMICAL', 'ORG',
             'ORDINAL', 'THERAPEUTIC_OR_PREVENTIVE_PROCEDURE',
             'DISEASE_OR_SYNDROME', 'CELL_COMPONENT', 'CHEMICAL',
             'DISEASE_OR_SYNDROME', 'GENE_OR_GENOME', 'CHEMICAL',
             'CARDINAL', 'CELL_COMPONENT', 'GENE_OR_GENOME',
             'GENE_OR_GENOME', 'CELL_COMPONENT', 'GENE_OR_GENOME', 'CHEMICAL',
             'GENE_OR_GENOME', 'GENE_OR_GENOME', 'CHEMICAL', 'GENE_OR_GENOME',
             'CARDINAL', 'GENE_OR_GENOME', 'CARDINAL', 'CORONAVIRUS',
             'GENE_OR_GENOME', 'GENE_OR_GENOME', 'CHEMICAL', 'FAC',
             'GENE_OR_GENOME', 'GENE_OR_GENOME', 'CELL_COMPONENT'],
            ['GENE_OR_GENOME', 'CELL', 'CELL_COMPONENT', 'ORG', 'PRODUCT',
             'CARDINAL', 'CELL_FUNCTION', 'GENE_OR_GENOME', 'CELL_COMPONENT',
             'FAC', 'GENE_OR_GENOME', 'CHEMICAL', 'CELL', 'GENE_OR_GENOME',
             'CHEMICAL', 'GENE_OR_GENOME', 'MOLECULAR_FUNCTION',
             'CELL_FUNCTION', 'CELL', 'GENE_OR_GENOME', 'CELL', 'CELL',
             'GENE_OR_GENOME', 'ORGANISM', 'CHEMICAL', 'CELL', 'CELL',
             'GENE_OR_GENOME', 'MOLECULAR_FUNCTION', 'CHEMICAL', 'CELL',
             'CHEMICAL', 'GENE_OR_GENOME', 'CELL', 'CELL_FUNCTION',
             'GENE_OR_GENOME', 'CELL_COMPONENT', 'GENE_OR_GENOME',
             'MOLECULAR_FUNCTION', 'CHEMICAL', 'DISEASE_OR_SYNDROME']]
        expected_link = [
            ['UMLS:C0206750', 'UMLS:C1175175', 'UMLS:C1425041',
             'UMLS:C0206750', 'UMLS:C4521602', 'UMLS:C1819995',
             'UMLS:C0282629', 'UMLS:C1279901', 'UMLS:C0947647',
             'UMLS:C3714514', 'UMLS:C0042760', 'UMLS:C2363430',
             'UMLS:C3714514', 'UMLS:C0037813', 'UMLS:C4521566',
             None, 'UMLS:C0042760', None, 'UMLS:C1706273', 'UMLS:C0042760',
             'UMLS:C0037813', 'UMLS:C0010076', None, 'UMLS:C1706273',
             'UMLS:C1514562', 'UMLS:C0014834', 'UMLS:C0205448',
             'UMLS:C1706273', 'UMLS:C5201140', 'UMLS:C1425041',
             None, None, 'UMLS:C0010654', None, None, 'UMLS:C1706273',
             'UMLS:C0042760'],
            ['UMLS:C0812198', 'UMLS:C1823878', 'UMLS:C4521340', None,
             'UMLS:C1823878', 'UMLS:C5201140', 'UMLS:C4759886',
             'UMLS:C0812198', 'UMLS:C0026237', 'UMLS:C1823878',
             'UMLS:C0812198', None, 'UMLS:C1823878', 'UMLS:C0812198',
             None, 'UMLS:C0812198', 'UMLS:C1519751', 'UMLS:C4759886',
             None, 'UMLS:C0812198', 'UMLS:C1823878', None,
             'UMLS:C0812198', None, 'UMLS:C2828366', 'UMLS:C1823878',
             None, 'UMLS:C0812198', 'UMLS:C1752727', 'UMLS:C0243077',
             'UMLS:C1823878', None, None, 'UMLS:C1823878',
             'UMLS:C4759886', 'UMLS:C0812198', 'UMLS:C0026237',
             'UMLS:C0812198', 'UMLS:C1752727', 'UMLS:C0243077',
             'UMLS:C3273930']]

        # Get processed pack from the dataset.
        for pack in self.nlp.process_dataset(self.dataset_path):
            doc_exists = True
            i = filename.index(pack.pack_name)
            for j, entity_link in enumerate(pack.get(MedicalEntityMention)):
                self.assertEqual(expected_text[i][j], entity_link.text)
                self.assertEqual(expected_type[i][j], entity_link.ner_type)
                self.assertEqual(expected_link[i][j], entity_link.umls_link)

        self.assertTrue(doc_exists)


if __name__ == '__main__':
    unittest.main()
