# Source code description
## Requirement:
Python 3.6
Pytorch 0.4.1

## Input format:
CoNLL format, with each character and its label split by a whitespace in a line. The "BMES" tag scheme is prefered.

别 O 

错 O

过 O

邻 O

近 O

大 B-LOC

鹏 M-LOC

湾 E-LOC

的 O

湿 O

地 O

## Pretrain embedding:
The pretrained embeddings(word embedding, char embedding and bichar embedding) are the same with [Lattice LSTM](https://www.aclweb.org/anthology/P18-1144)

## Run the code:
1. Download the character embeddings and word embeddings from [Lattice LSTM](https://github.com/jiesutd/LatticeLSTM) and put them in the `data` folder.
2. To train on the datasets:
`python main.py `
3. To train/test your own data: modify the command with your file path and run.

