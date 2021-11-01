# Reinforcement Learning for Test Case Prioritization: Online Appendix
This reposiroty contains the traning scripts, datasets, and experiments corrpsonding to our paper [Reinforcement Learning for Test Case Prioritization.](https://arxiv.org/pdf/2011.01834.pdf)

## Datasets. 
The papers uses eight datasets that are placed in folder `data`. [Paint-Control](https://github.com/moji1/tp_rl/blob/master/data/tc_data_paintcontrol.csv) and [IOFROL](https://github.com/moji1/tp_rl/blob/master/data/iofrol.csv) are simple datasets that contains only features related to the execution history of test cases. Simple datasets are adapoted from [previous work](https://www.simula.no/sites/default/files/publications/files/reinforcement_learning_for_test_case_prioritization-issta17_0.pdf) and the detailed description of the datasets can be found [here](https://bitbucket.org/HelgeS/retecs/src/master/DATA/).
We refer to the rest of datasets (e.g., [Codec](https://github.com/moji1/tp_rl/blob/master/data/Commons_codec.csv)) as enriched datasets since they include extra features related to the complexity metrics of the test cases' code. Enrcihed dataset are adopted from [previous work](https://cin.ufpe.br/~bafm/publications/bertolino_etal_icse20.pdf). Table 1, Table 3, and Section 5.1 in our [paper](https://arxiv.org/pdf/2011.01834.pdf) provide detailed information concerning enrchied datasets. 

## Experiments
The paper evalutes seven RL algorithms (A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO) for training of an RL agent capable of Test Case Prioritization (TCP) based on three ranking apporches (pairwise, pointwise, and listwise). Folder `experiments` contains the result of the experiments that are repopted by the paper. For example, file `experiments/exp_result/PAIRWISE/A2C/Commons_codec_4.csv` shows how an RL agent that uses pairwise raking and is trained using A2C algorithm performs TCP for `Commons_codec` subjet. Also the digit appended to the name of the subjets refer to the `hisoty window` that shows how .....

## Rerun the experiments
To run the experiments, you first need to install Python 3.7 and Stable Baselines2 ("2.10"). You can then clone the repository and resrun the epecriment using `python testCase_prioritization/TPDRL.py` by passing the following options:

```
 -m MODE --> ranking model that can be either pairwise, pointwise, or listwise
 -a ALGO --> The RL algorithm to train the agent that can be either A2C, ACER, ACKTR, DQN, PPO1, PPO2, or TRPO
 -t TRAIN_DATA --> Location of the training dataset
  -e EPISODES. --> the number of episodes used to traing the agent
 [-d DATASET_TYPE]  --> type of the dataset that can be either simple or enrcihed
 [-w WIN_SIZE]      --> A number that specfies how ....
 [-f FIRST_CYCLE]   --> a cycle from which the training will be started. Each dataset contains many cycles of regression testing and this paramater specefies the number of cycles logs based on which the initial agent will be trained.
 [-c CYCLE_COUNT] --> How many cycles data will be given to agent at each training. 
 [-o OUTPUT_PATH] --> The path where the results and the trained agents will be saved
```

### An Example.
`python testCase_prioritization/TPDRL.py -m pointwise -a ACKTR -t ../data/iofrol-additional-features.csv -e 200 -w 10 -d simple`

Train an EL agent using ACKTR algorithm based on pointwise ranking for iofrol dataset. 






