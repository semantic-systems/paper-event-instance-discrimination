# Submission@Text2Story@ECIR24: Event Extraction Alone Is Not Enough


# De-noising Procedures
The crawled gdelt news are noisy, 
which is natural in the regex-based keyword search used in gdelt. 
Examples of noisy instances are followed:
   1. Image captions of a link in the main page mentioning the search keywords
   2. Usage of the search keyword appears in context with pragmatical difference,
      - A flood of doubts arrives at the White House.
      - An Earthquaking speech by president Trump.

## How to detect and remove them
1. S-bert based clustering (min_cluster_size=50, min_similarity=0.7)
2. annotate event type with an event type detector that is trained to detect tropical storm events, which in this benchmark is considered as flooding, hurricane, tornado, tsunami and (tropical) storm. Other events include earthquakes, explosions, wildfire and drought.
3. annotate linked entities with spacy linker,
4. remove clusters whose majority prediction is not storm (benchmark bias 1: false negatives of event detector)
5. temporal clustering: 1-D DBSCAN clustering on publication date (min_samples = 3, delta = 1). Remove outliers. Keep the biggest cluster. All instances in a cluster are published continuously (one-day interval). 
6. merge clusters by temporal consistency and entity overlap. 

# Where are the data?
## Downloading
### Eventist
- unsplit: https://drive.google.com/file/d/13tMtLuU3_wprb4bOIdxq_OJmaKRvGqoY/view?usp=sharing
- train: https://drive.google.com/file/d/1G3LdoM58ZCY8SlhwUdD-Vc_gbFDRijC5/view?usp=sharing
- valid: https://drive.google.com/file/d/1FZn6Vc4B6BqtyJgexiRegv3nO6uacX_D/view?usp=sharing
- test: https://drive.google.com/file/d/1_HZEPC0n9B8JnWfYO6jzT1S-LXv5a_Mg/view?usp=sharing

### Crisisfacts
Additionally, crisisfacts data is also provided, 
- unsplit:https://drive.google.com/file/d/1n9n0CFQRMT3v3NPvsKXQbuxzhgOJti7g/view?usp=sharing
- train: https://drive.google.com/file/d/11B6Hqe1QaJRqlPezZ0ugOnMWr0-4TA1L/view?usp=sharing
- valid:https://drive.google.com/file/d/1RytVYmFEQweDDufUCll8vm4SbxmfyaWE/view?usp=sharing
- test: https://drive.google.com/file/d/1Cs39sO8hIB6FTB5CLiInRycbNNPC0gGv/view?usp=sharing




For Eventist, the final splits should be stored here:
`./data/stormy_data/final_df_v3.csv `
`./data/stormy_data/train_v3.csv `
`./data/stormy_data/valid_v3.csv `
`./data/stormy_data/test_v3.csv `

Additionally, crisisfacts data splits should be stored here:
`./data/stormy_data/test_from_crisisfacts.csv `
`./data/crisisfacts_data/train_v3.csv `
`./data/crisisfacts_data/valid_v3.csv `
`./data/crisisfacts_data/test_v3.csv `

# How to train?
`python ./models/EventPairwiseTemporality.py`

(Evaluation will be done after training)