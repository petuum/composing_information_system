# Copyright 2019 The Forte Authors. All Rights Reserved.
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
This is a demo pipeline showed how to build a qa engine with nlp tools.
"""
import os
import yaml
import torch

from forte.common.configuration import Config
from forte.data.multi_pack import MultiPack
from forte.data.readers import MultiPackTerminalReader
from forte.pipeline import Pipeline
from forte.data.selector import NameMatchSelector, RegexNameMatchSelector
from forte_wrapper.allennlp.allennlp_processors import AllenNLPProcessor
from forte_wrapper.nltk.nltk_processors import NLTKLemmatizer, \
    NLTKSentenceSegmenter, NLTKWordTokenizer, NLTKPOSTagger
from forte_wrapper.elastic.elastic_search_processor import \
    ElasticSearchProcessor

from composable_source.processors.elasticsearch_query_creator import \
    ElasticSearchQueryCreator
from composable_source.processors.scispacy_processor import SciSpacyProcessor
from composable_source.processors.response_creator import ResponseCreator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = yaml.safe_load(open(config_file, "r"))
    config = Config(config, default_hparams=None)

    # build pipeline
    nlp: Pipeline[MultiPack] = Pipeline()
    nlp.set_reader(reader=MultiPackTerminalReader(), config=config.reader)

    # process input and generate query
    selector_input = NameMatchSelector(select_name=config.reader.pack_name)
    nlp.add(NLTKSentenceSegmenter(), selector=selector_input)
    nlp.add(NLTKWordTokenizer(), selector=selector_input)
    nlp.add(NLTKPOSTagger(), selector=selector_input)
    nlp.add(NLTKLemmatizer(), selector=selector_input)
    nlp.add(AllenNLPProcessor(),
            config=config.allennlp_query, selector=selector_input)

    nlp.add(ElasticSearchQueryCreator(), config=config.query_creator)

    # search
    nlp.add(ElasticSearchProcessor(), config=config.indexer)

    # process hits
    pattern = rf"{config.indexer.response_pack_name_prefix}_\d"
    selector_hit = RegexNameMatchSelector(select_name=pattern)
    nlp.add(component=SciSpacyProcessor(),
            config=config.spacy1, selector=selector_hit)
    nlp.add(component=SciSpacyProcessor(),
            config=config.spacy2, selector=selector_hit)
    nlp.add(AllenNLPProcessor(), config=config.allennlp, selector=selector_hit)
    nlp.add(NLTKPOSTagger(), selector=selector_hit)
    nlp.add(NLTKLemmatizer(), selector=selector_hit)

    # generate outputs
    nlp.add(ResponseCreator(), config=config.response)

    nlp.initialize()

    # process dataset
    m_pack: MultiPack
    for m_pack in nlp.process_dataset():
        print('The number of datapacks(including query) is', len(m_pack.packs))
        if len(m_pack.packs) == 1:  # no paper found, only query
            input("No result ...\n")
            continue

    print('Done')
