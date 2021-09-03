import unittest
from examples.pipeline.inference.search_cord19 import *


class BuildSearchPipelineTest(unittest.TestCase):
    def test_pipeline(self):
        config_file = "sample_data/tests/config.yml"
        config = yaml.safe_load(open(config_file, "r"))
        config = Config(config, default_hparams=None)
        build_search_pipeline(config)
