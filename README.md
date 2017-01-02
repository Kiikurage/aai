# Team G 引き分けオセロ

## Background && Motivation

- Evolution of Game AI
- Can we create AI that can lead to draw?


## Requirements 

- Python &ge; 3.4
- Chainer &ge; 1.5

## Supervised Learning from Kifu

```python
python training/supervised_learning.py [-g GPU] [--out OUT_DIR] [--small] [--use_bn]
```


## Components

- 高速なオセロシミュレータ

- 盤面を入力とし、勝率を高精度に予測する評価関数

- 相手の実力を加味したうえでの方策

## Reference

- オセロAIの有名論文 Experiments: with Multi-ProbCut and a New High-Quality Evaluation Function for Othello
- 強化学習オセロの有名論文 MOUSE(μ): A self-teaching algorithm that achieved master-strength at Othello 
- AlphaGo : [Mastering the game of Go with deep neural networks and tree search](http://www.nature.com/nature/journal/v529/n7587/pdf/nature16961.pdf)


## Cのモジュールについて

`python setup.py build_ext --inplace` でビルドできます。