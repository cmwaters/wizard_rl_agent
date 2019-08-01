# Wizard RL Agent

In this project, reinforcement learning is applied to the card game Wizard. We present meRLin, an agent with a two 
phase architecture that uses Proximal Policy Optimization for playing cards and a neural network for making trick 
predictions. Our proposed agent significantly outperforms both rule based and other reinforcement learning approaches. 
Evaluation against human players shows that meRLin is often able to win in the first few games, however human players 
are able to adapt fast to its playstyle and exploit it to win most of the remaining games.

#### Setup

Install dependencies: \
```pip install -r requirements.txt```

#### Usage

To play against the agent using the GUI: \
```$ cd wizard_site``` \
```$ python3 manage.py runserver```

From the root directory, to train and evaluate RL agent: \
```$ cd tests``` \
```$ python3 test_train_and_evaluate.py```

To update the requirements.txt file: \
```$ pip freeze > requirements.txt```

#### Wizard Wikipedia Page:

https://en.wikipedia.org/wiki/Wizard_(card_game) 

#### Git Commands
https://github.com/joshnh/Git-Commands

#### Style Guide

https://www.python.org/dev/peps/pep-0008/

https://google.github.io/styleguide/pyguide.html

#### Markdown CheatSheet

https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#code