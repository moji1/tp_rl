# Reinforcement Learning for Test Case Prioritization: Online Appendix
This reposiroty contains the traning scripts, datasets, and experiments corrpsonding to our paper [Reinforcement Learning for Test Case Prioritization.](https://arxiv.org/pdf/2011.01834.pdf)

## Datasets. 
The papers uses eight datasets that are placed in folder `data`. [Paint-Control](https://github.com/moji1/tp_rl/blob/master/data/tc_data_paintcontrol.csv) and [IOFROL](https://github.com/moji1/tp_rl/blob/master/data/iofrol.csv) are simple datasets that contains only features related to the execution history of test cases. Simple datasets are adapoted from [previous work](https://www.simula.no/sites/default/files/publications/files/reinforcement_learning_for_test_case_prioritization-issta17_0.pdf) and the detailed description of the datasets can be found [here](https://bitbucket.org/HelgeS/retecs/src/master/DATA/).
We refer to the rest of datasets (e.g., [Codec](https://github.com/moji1/tp_rl/blob/master/data/Commons_codec.csv)) as enriched datasets since they include extra features related to the complexity metrics of the test cases' code. Enrcihed dataset are adopted from [previous work](https://cin.ufpe.br/~bafm/publications/bertolino_etal_icse20.pdf). Table 1, Table 3, and Section 5.1 in our [paper](https://arxiv.org/pdf/2011.01834.pdf) provide detailed information concerning enrchied datasets.  


