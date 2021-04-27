# Composing Information System - A Q&amp;A Engine

The repository shows how to build a Q&amp;A engine using Forte and existing NLP models, and allows users to quickly re-purpose it for different datasets and/or tasks. 

The current showcase contains two pipelines: data index pipeline, and Q&A pipeline.

## Dataset

We used COVID-19 Open Research Dataset Challenge (CORD-19) Dataset. It contains thousands of scientific and medical papers from the National Institutes of Health for COVID-19.

Link: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge


## Task
Given user input query, the pipeline will search the relevant information in CORD-19 dataset, and return answers which contains relation, sentence, as well as relevant medical concepts.


## How to run the pipeline
First, you need to create a virtual environment and git clone this repo, then in command line:

`pip install -r requirements.txt`

If you don't have Cython installed, you will need to `pip install Cython` first to avoid installation failure.


### Build index
* Prerequisite: ElasticSearch

Please follow https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html to install ElasticSearch.

Then we need to have ElasticSearch running on backend. Do

`cd elasticsearch-7.11.1`
`./bin/elasticsearch`

to start the server.

* Build Elasticsearch Indexer:

For CORD-19 dataset we used the data in json format for index, which is in `document_parses/pdf_json`.

You can change the config of ElasticSearch in `examples/pipeline/indexer/config.yml`. Then run

`python examples/pipeline/indexer/cordindexer.py --data-dir [your_data_directory]`
    
to index the files in `your_data_directory`. 



### Build QA engine
* Prerequisites: StanfordNLP 
Please install following the instructions on official Link: https://stanfordnlp.github.io/CoreNLP/download.html.

Then we need to have StanfordNLP server running on backend, you can run

`cd stanford-corenlp-4.2.0`

`java -cp "*" -mx3g edu.stanford.nlp.pipeline.StanfordCoreNLPServer`

to start the server.

    
* Run QA pipeline

The QA pipeline uses Scispacy models for entity linking, and AllenNLP models for SRL and OpenIE task.

You can run

`python examples/pipeline/inference/search_cord19.py`

to get started. It will take a while to initialize the pipeline.


When you see `Enter your query here:`, you can start to ask questions, such as
```
'what does covid-19 cause', 
'what does covid-19 affect', 
'what caused liver injury', 
'what caused renal involvement'
```

The pipeline will process the query, then search the relations and output the result to human readable format.

Here is an example of the question 'what does covid-19 cause'. The output shows the potential relation, the source sentence and source article that the relation comes from, 
as well as the relevant UMLS medical concepts in the relations.

```
•Relation:
COVID-19	causes	infection in the pulmonary system
•Source Sentence:
COVID-19 enters the cell by penetrating ACE2 at low cytosolic pH and causes infection in the pulmonary system [2, 3] .(From Paper: , Can COVID-19 cause myalgia with a completely different mechanism? A hypothesis)
•UMLS Concepts:
 - covid-19
	Name: COVID-19	CUI: C5203670	Learn more at: https://www.ncbi.nlm.nih.gov/search/all/?term=C5203670
================================================================================
```

## Models

The pipeline contains three major steps: __Query Understanding, Document Retrieval, and Answer Extraction__.

### Query Understanding
Forte analyzes user’s input question by annotating the basic language features using __NLTK__ word tokenizer, POS tagger and lemmatizer. 
Then __AllenNLP’s SRE model__ was utilized to extract the arguments and predicate in the question, and which argument that the user is interested in. 
This annotated question is transformed into a query and pass to the next step. 


### Document Retrieval
The __ElasticSearch__ backend is fast and quickly filters information from a large corpus, 
which is great for larger corpora with millions of documents. It is utilized as the search engine here. 

The query created in the last step was used to retrieve relevant articles from an index that’s stored in ElasticSearch. You could set the number of retrieved documents in `config.yml`.

The extracted documents was stored as datapack in Forte, and passed to the next step to generate final output.


### Answer Extraction
Given relevant document datapacks, the system helps to extract the relevant relations. 

Here, __ScispaCy models__ trained on biomedical text were utilized to do __sentence parsing__, __NER__, and __entity linking__ with UMLS concepts. 
__AllenNLP’s OpenIE model__ was utilized for __relation extraction__. 
__NLTK__ Lemmatizer was also used to process predicate of the relations.

Given all relations, the relation that matched user's query will be selected as candidate answer, which means the predicate lemma in the relation and user query's predicate lemma are the same, and the user query's argument was mentioned in the extracted relation.

Finally, the relations were polished by adding references and supporting information that’s useful for fact-checking and countering misinformation.
So the source sentence and article were also provided in the result. 
Besides, some terms in the relation could be linked to UMLS concepts, which brings standards for further interoperability or research, so they were also listed on the result.

The final result was organized to three parts: __Relation, Source Sentence, and UMLS Concepts__ for reading purpose. 


