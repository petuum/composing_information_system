from ft.onto.base_ontology import Token, Sentence, PredicateLink
from forte.data.data_pack import DataPack
from collections import defaultdict


def query_preprocess(input_pack: DataPack):
    """
    Extract nouns and verb from user input query.
    :param input_pack:
    :return:sentence: query text
        arg0: subject in query
        arg1: object in query
        predicate: verb in query
        verb_lemma: verb lemma
        is_answer_arg0: should subject(arg0) or object(arg1) be returned as answer
    """
    sentence = input_pack.get_single(Sentence)

    relations = defaultdict(dict)
    text_mention_mapping = {}

    # get all srl relations
    for link in input_pack.get(PredicateLink, sentence):
        verb = link.get_parent()
        verb_text = verb.text
        argument = link.get_child()
        argument_text = argument.text

        text_mention_mapping[verb_text] = verb
        text_mention_mapping[argument_text] = argument
        relations[verb_text][link.arg_type] = argument_text

    arg0, arg1, predicate = None, None, None
    for verb_text, entity in relations.items():
        arg0, arg1, predicate = collect_mentions(text_mention_mapping, entity, verb_text)
        if not arg0 and not arg1:
            continue
        else:
            break

    if not arg0 and not arg1:
        raise Exception('AllenNLP SRL cannot extract the two arguments or the '
                        'predicate in your query, please check our examples '
                        'or rephrase your question')

    verb_lemma, is_answer_arg0 = None, None

    # check pos tag and lemma for tokens
    for j, token in enumerate(input_pack.get(entry_type=Token,
                                             range_annotation=sentence,
                                             components=['forte_wrapper.nltk.nltk_processors.NLTKWordTokenizer']
                                             )):
        # find WH words
        if token.pos in {"WP", "WP$", "WRB", "WDT"}:
            if arg0.begin <= token.begin and arg0.end >= token.end:
                is_answer_arg0 = True
            elif arg1.begin <= token.begin and arg1.end >= token.end:
                is_answer_arg0 = False

        # find verb lemma
        if token.text == predicate.text:
            verb_lemma = token.lemma

    return sentence, arg0.text if arg0 else '', arg1.text if arg1 else '', \
           predicate.text, verb_lemma, is_answer_arg0


def collect_mentions(text_mention_mapping, relation, verb_text):
    """
    Get arg0,arg1 and predicate entity mention
    :param text_mention_mapping:
    :param relation:
    :param verb_text:
    :return:
    """
    arg0_text, arg1_text = get_arg_text(relation)

    if not arg0_text or not arg1_text:
        return None, None, None

    arg0 = text_mention_mapping[arg0_text]
    arg1 = text_mention_mapping[arg1_text]
    predicate = text_mention_mapping[verb_text]

    return arg0, arg1, predicate


def get_arg_text(relation):
    """
    find arg0 and arg1 text in all relations. we considered 3 annotation
    for comprehensive subject and object extraction
    As AllenNLP uses PropBank Annotation, each verb sense has numbered
    arguments e.g., ARG-0, ARG-1, etc.
    ARG-0 is usually PROTO-AGENT
    ARG-1 is usually PROTO-PATIENT
    ARG-2 is usually benefactive, instrument, attribute
    :param relation:
    :return:
    """
    arg0_text, arg1_text = None, None
    if 'ARG0' in relation and 'ARG1' in relation:
        arg0_text = relation['ARG0']
        arg1_text = relation['ARG1']

    elif 'ARG1' in relation and 'ARG2' in relation:
        arg0_text = relation['ARG1']
        arg1_text = relation['ARG2']

    return arg0_text, arg1_text
