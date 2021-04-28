# Composable Training Pipeline

The repo contains four training pipelines: general NER, bio-medical NER,
wiki entity linking, and medical entity linking.

## Tasks and Datasets

**General NER**

* Task: NER in general domain.

* Model: General BERT.

* Dataset for model training :   
    English data from CoNLL 2003 shared task. It contains four different types of named entities: PERSON, LOCATION, ORGANIZATION, and MISC.
    It can be downloaded from [DeepAI website](https://deepai.org/dataset/conll-2003-english).

* Use Case:  ​Analyzing research papers on COVID-19​

    * Materials to use for testing:  ​Research Papaers for COVID-19.​

    * Model: BioBERT v1.1 (domain-specific language representation model pre-trained on large-scale biomedical corpora)​

    * Dataset for model training: 
    NER: CORD-NER dataset,  Entity-Linking: CORD-NERD Dataset​
​

**Bio-medical NER**

* Task: NER in bio-medical domain.

* Model: BioBERT v1.1 (domain-specific language representation model pre-trained on large-scale biomedical corpora).

* Dataset for model training :   
    MTL-Bioinformatics-2016. Download and know more about this dataset on 
    [this github repo](https://github.com/cambridgeltl/MTL-Bioinformatics-2016). 


**Wiki entity linking**

* Task: Entity linking in general domain.

* Model: General Bert.

* Dataset for model training : 
    AIDA CoNLL03 entity linking dataset. The entities are identified by YAGO2 entity name, by Wikipedia URL, or by Freebase mid.
    It has to be used together with CoNLL03 NER dataset mentioned above. 
    
    First download CoNLL03 dataset which contains train/dev/test.
    
    Second, download aida-yago2-dataset.zip from [this website](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads).
    
    Third, in the downloaded folder, manually segment AIDA-YAGO2-annotations.tsv into three files corresponding to CoNLL03 train/dev/test,
    then put them into train/dev/test folders.


**Medical entity linking**

* Task: Entity linking in medical domain.

* Model: BioBERT v1.1 (domain-specific language representation model pre-trained on large-scale biomedical corpora).

* Dataset for model training: 
    MedMentions st21pv sub-dataset. It can be download from [this github repo](https://github.com/chanzuckerberg/MedMentions/tree/master/st21pv).


## How to train your models

* Below shows the steps to train a general NER model. You can modify config direcotry to train others.

* Create a conda virtual environment and git clone this repo and go into the repo, then in command line:

    `cd composing_information_system/`

    `export PYTHONPATH="$(pwd):$PYTHONPATH"`

* Create an output directory.

    `mkdir sample_output`

* Run training script:

    `python examples/tagging/main_train_tagging.py --config-dir examples/tagging/configs_ner/`

* After training, you will find your trained models in the following directory. It contains the trained model, vocabulary, train state and training log. 

    `ls sample_output/ner/`

    

## How to do inference

* Download the pretrained models and vocabs from below list. Put model.pt and vocab.pkl into `predict_path` specified in `config_predict.yml`.
Then run the following command.

    `python examples/tagging/main_predict_tagging.py --config-dir examples/tagging/configs_ner/`

    + General Ner : [__model__](https://drive.google.com/file/d/1WCSwDw8WEjshf1IUY4iMPQBV_HyhuwNm/view?usp=sharing), 
    [__train_state__](https://drive.google.com/file/d/1cDmDNFDZLgZLr2BO4MeuMjD_eh4T3PQI/view?usp=sharing), 
    [__vocab__](https://drive.google.com/file/d/18UFHFg9gfZbb9Sb5s8_h0eG2sd9jXn4H/view?usp=sharing)

    + Bio-medical NER : [__model__](https://drive.google.com/file/d/1dL2PYSPb-HOiSBQeQiVD530uxmLGbfbP/view?usp=sharing), 
    [__train_state__](https://drive.google.com/file/d/1bJq1RUGK1h3epjklFZEMGLY34SEO9Svh/view?usp=sharing), 
    [__vocab__](https://drive.google.com/file/d/1yhQriZjABv3XA_0I4w9jD8k3n3SFvJqc/view?usp=sharing)

    + Wiki entity linking : [__model__](https://drive.google.com/file/d/1injnv7s8a-PAhfwMN25kyzxh-FGwnWNZ/view?usp=sharing), 
    [__train_state__](https://drive.google.com/file/d/1pttk34Fk3fWJz-Vfy3kCY8ET7qReJH84/view?usp=sharing), 
    [__vocab__](https`://drive.google.com/file/d/19OVGetDQ7BJ1m1_FyxkDE2SI266XvNLv/view?usp=sharing)
    
    + Medical entity linking : [__model__](https://drive.google.com/file/d/1kBDItWrguZez0F57xmT90eHEucc2CjE-/view?usp=sharing), 
    [__train_state__](https://drive.google.com/file/d/1qbL7gb3SMvgHUMhuD_-tQFFvXYBvV6Pl/view?usp=sharing), 
    [__vocab__](https://drive.google.com/file/d/1UrcIF3ZwllWdee0wEbCYpZ6x9bGXwEdy/view?usp=sharing)



### Inferencing using two concatenated models

* You can use your trained bio-medical NER model and medical entity linking model to do inference on a new dataset

* Inference Dataset :
    CORD-NERD dataset. Information about this dataset and downloadble links can be found [here](https://aistairc.github.io/BENNERD/).

    `python examples/tagging/main_predict_cord.py --config-dir examples/tagging/configs_cord_test/`


## Eveluation performance examples:

* Below is the performance metrics of the General NER task.

    
    |       General NER          |                                                       |
    |----------------------------|-------------------------------------------------------|
    |   Overall         |accuracy:  98.98% precision:  93.56%; recall:  94.81%; FB1:  94.18|
    |   LOC             | precision:  95.94%; recall:  96.41%; FB1:  96.17  18430|
    |   MISC            | precision:  86.15%; recall:  89.97%; FB1:  88.02  9628|
    |   ORG             | precision:  91.05%; recall:  91.99%; FB1:  91.52  13549|
    |   PER             | precision:  96.89%; recall:  97.69%; FB1:  97.29  18563|



* Below is the performance metrics of the bio-medical NER task.

    
    |       Bio-medical NER          |                                                      |
    |--------------------------------|-------------------------------------------------------|
    |   Overall         |accuracy:  98.41%; precision:  84.93%; recall:  89.01%; FB1:  86.92|
    |   Chemical        | precision:  79.20%; recall:  86.34%; FB1:  82.62  1428|
    |   Organism        | precision:  85.23%; recall:  73.87%; FB1:  79.14  3337|
    |   Protein         | precision:  85.53%; recall:  97.15%; FB1:  90.97  11972|
  

* Below is the performance metrics of the wiki entity linking task. 
Due to the large number of classes in entity linking tasks, we are only showing the overall performance.

    |   Wiki entity linking          |                                                        |
    |---------------------------------|-------------------------------------------------------|
    |        Overall             | accuracy:  91.27%; precision:  51.86%; recall:  38.60%; FB1:  44.25|
    
* Below is the performance metrics of the medical entity linking task. 
Since MedMentions dataset does not provide word boundaries (only has entity linking boundaries),
the evaluation method here is to count extact match of entities.

    |   Medical entity linking         |                                                        |
    |---------------------------------|-------------------------------------------------------|
    |       Exact match             |   precision:      26.25%;     recall:     22.24%;     f1:     24.07%     |
