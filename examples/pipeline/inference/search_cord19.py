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

import os

import torch
import yaml
from forte.common.configuration import Config
from forte.data.caster import MultiPackBoxer
from forte.data.multi_pack import MultiPack
from forte.data.readers import TerminalReader
from forte.pipeline import Pipeline
from forte.data.selector import RegexNameMatchSelector
from forte.spacy.spacy_processors import SpacyProcessor
from forte.allennlp import AllenNLPProcessor
from forte.elastic import ElasticSearchProcessor
from forte.nltk import (
    NLTKLemmatizer,
    NLTKSentenceSegmenter,
    NLTKWordTokenizer,
    NLTKPOSTagger,
)

from composable_source.processors.elasticsearch_query_creator import (
    ElasticSearchQueryCreator,
)
from composable_source.processors.response_creator import ResponseCreator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_search_pipeline(config: Config):
    # Build pipeline and add the reader, which will read from terminal.
    nlp: Pipeline = Pipeline()
    nlp.set_reader(reader=TerminalReader())

    # Conduct query analysis.
    nlp.add(NLTKSentenceSegmenter())
    nlp.add(NLTKWordTokenizer())
    nlp.add(NLTKPOSTagger())
    nlp.add(NLTKLemmatizer())
    nlp.add(AllenNLPProcessor(), config=config.allennlp_query)

    # Start to work on multi-packs in the rest of the pipeline, so we use a
    # boxer to change this.
    nlp.add(MultiPackBoxer(), config=config.boxer)

    # Create query.
    nlp.add(ElasticSearchQueryCreator(), config=config.query_creator)

    # Search the elastic back end.
    nlp.add(ElasticSearchProcessor(), config=config.indexer)

    # process hits
    pattern = rf"{config.indexer.response_pack_name_prefix}_\d"
    selector_hit = RegexNameMatchSelector(select_name=pattern)
    nlp.add(
        component=SpacyProcessor(), config=config.spacy1, selector=selector_hit
    )
    nlp.add(
        component=SpacyProcessor(), config=config.spacy2, selector=selector_hit
    )
    nlp.add(AllenNLPProcessor(), config=config.allennlp, selector=selector_hit)
    nlp.add(NLTKPOSTagger(), selector=selector_hit)
    nlp.add(NLTKLemmatizer(), selector=selector_hit)

    # generate outputs
    nlp.add(ResponseCreator(), config=config.response)

    return nlp


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "config.yml")
    config = yaml.safe_load(open(config_file, "r"))
    config = Config(config, default_hparams=None)
    nlp = build_search_pipeline(config)
    nlp.initialize()

    # process dataset
    m_pack: MultiPack
    for m_pack in nlp.process_dataset():
        print("The number of datapacks(including query) is", len(m_pack.packs))
        if len(m_pack.packs) == 1:  # no paper found, only query
            input("No result. Try another query: \n")
            continue

    print("Done")
