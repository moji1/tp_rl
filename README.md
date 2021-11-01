# Reinforcement Learning for Test Case Prioritization: Online Appendix
This repository contains the training scripts, datasets, and experiments corresponding to our paper [Reinforcement Learning for Test Case Prioritization.](https://arxiv.org/pdf/2011.01834.pdf)

## Datasets. 
The paper uses eight datasets that are placed in folder `data`. [Paint-Control](https://github.com/moji1/tp_rl/blob/master/data/tc_data_paintcontrol.csv) and [IOFROL](https://github.com/moji1/tp_rl/blob/master/data/iofrol.csv) are simple datasets that contain only features related to the execution history of test cases. Simple datasets are adopoted from [previous work](https://www.simula.no/sites/default/files/publications/files/reinforcement_learning_for_test_case_prioritization-issta17_0.pdf), and the detailed description of the datasets can be found [here](https://bitbucket.org/HelgeS/retecs/src/master/DATA/).
We refer to the rest of the datasets (e.g., [Codec](https://github.com/moji1/tp_rl/blob/master/data/Commons_codec.csv)) as enriched datasets since they include extra features related to the complexity metrics of the test cases' code. Enriched datasets are adopted from [previous work](https://cin.ufpe.br/~bafm/publications/bertolino_etal_icse20.pdf). Table 1, Table 3, and Section 5.1 in our [paper](https://arxiv.org/pdf/2011.01834.pdf) provide detailed information concerning enriched datasets. 

## Experiments
The paper evaluates seven RL algorithms (A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO) for the training of an RL agent capable of Test Case Prioritization (TCP) based on three ranking approaches (pairwise, pointwise, and listwise). Folder `experiments` contains the result of the experiments that are reported by the paper. For example, file `experiments/exp_result/PAIRWISE/A2C/Commons_codec_4.csv` shows how an RL agent that uses pairwise raking and is trained using A2C algorithm performs TCP for `Commons_codec` subjet. Also, the digit (4) that is appended to the name of the subjets refers to the `history window` that specifies how many of the previous execution results of test cases to be included in the calculation of features such as average execution time or failure rate. 

## Rerun the experiments
To run the experiments, you first need to install Python 3.7 and Stable Baselines2 ("2.10"). You can then clone the repository and rerun the experiment using `python testCase_prioritization/TPDRL.py` by passing the following options:

```
 -m MODE --> ranking model that can be either pairwise, pointwise, or listwise
 -a ALGO --> The RL algorithm to train the agent that can be either A2C, ACER, ACKTR, DQN, PPO1, PPO2, or TRPO
 -t TRAIN_DATA --> Location of the training dataset
  -e EPISODES. --> the number of episodes used to train the agent
 [-d DATASET_TYPE]  --> type of the dataset that can be either simple or enriched
 [-w WIN_SIZE]      --> history window
 [-f FIRST_CYCLE]   --> a cycle from which the training will be started. Each dataset contains many cycles of regression testing, and this parameter specifies the number of cycles whose logs are to be used to train the initial agent.
 [-c CYCLE_COUNT] --> How frequently the agent will be retrained, e.g., one means that the agent will be retrained after each CI cycle.
 [-o OUTPUT_PATH] --> The path where the results and the trained agents will be saved
```

### An Example.
`python testCase_prioritization/TPDRL.py -m pointwise -a ACKTR -t ../data/iofrol-additional-features.csv -e 200 -w 10 -d simple`

The above command starts training of an RL agent using ACKTR algorithm and pointwise ranking based on dataset `iofrol`. 






