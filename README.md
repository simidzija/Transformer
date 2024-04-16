# Transformer

This repo implements a decoder-only transformer in pytorch. 
We use only the built in `nn.Module` and `nn.Parameter` classes, implementing all layer classes from scratch:
- [layers.py](layers.py): implementation of all layer classes, including the `Transformer` class
- [nlp.py](nlp.py): natural language processing tools

We then train our transformer model on data:
- [demo_1.ipynb](demo_1.ipynb): train Transformer to simply copy an integer sequence
- [demo_2.ipynb](demo_2.ipynb): train Transformer to reverse an integer sequence
- [demo_3.ipynb](demo_3.ipynb): train Transformer on Beatles lyrics
