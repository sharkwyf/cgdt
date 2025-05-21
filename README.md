# Critic-Guided Decision Transformer for Offline Reinforcement Learning
This repository contains the Pytorch implementation of [Critic-Guided Decision Transformer for Offline Reinforcement Learning](https://arxiv.org/abs/2312.13716).

## Requirements
```console
conda env create -f conda_env.yml
source activate cgdt
```

## Example
To train a CGDT agent for `hopper` with the `medium-v2` dataset:
```console
python main.py
```

If you use this code for your research, please cite us as:
```Bibtex
@inproceedings{wang2024critic,
  title={Critic-guided decision transformer for offline reinforcement learning},
  author={Wang, Yuanfu and Yang, Chao and Wen, Ying and Liu, Yu and Qiao, Yu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={14},
  pages={15706--15714},
  year={2024}
}
```

## License
The majority of `cgdt` is licensed under MIT License, however portions of the project are available under separate license terms: 
* online-dt - CC-BY-NC
* D4RL dataset -  Creative Commons Attribution 4.0 License (CC-BY)
* D4RL code, transformers, Lamb - Apache 2.0 License
* stable-baselines3, Gym, decision-transformer - MIT License

