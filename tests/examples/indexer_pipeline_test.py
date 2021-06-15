import unittest
from examples.pipeline.indexer.cordindexer import *


class BuildIndexPipelineTest(unittest.TestCase):
    def test_pipeline(self):
        config_file = 'sample_data/tests/config.yml'
        config = yaml.safe_load(open(config_file, "r"))
        config = Config(config, default_hparams=None)

        data_dir = 'sample_data/tests/cord19research'
        build_index_pipeline(data_dir, config)
