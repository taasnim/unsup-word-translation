# Unsupervised Word Translation

This repository contains the code of our paper *Revisiting Adversarial Autoencoder for Unsupervised Word Translation with Cycle Consistency and Improved Training*  (NAACL-HLT 2019).


## Requirements

- Python 3 with NumPy/SciPy
- PyTorch 0.4 or higher
- tqdm


## Get Datasets

### Get Monolingual Word Embeddings

All the monolingual word embeddings should be in the `'./data' folder`. </br>
[FastText Embeddings](https://fasttext.cc/docs/en/pretrained-vectors.html): You can download the English (en) and Spanish (es) embeddings this way:
```bash
# English fastText Wikipedia embeddings
cd data/
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
# Spanish fastText Wikipedia embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.es.vec
```

### Get evaluation datasets

Download [bilingual dictionaries](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries) from MUSE project.
You can simply run:
```bash
cd data/
./get_evaluation.sh
```

For downloading Dinu-Artetxe dataset please visit [vecmap](https://github.com/artetxem/vecmap/) repository.


## How to run

The general command:
CUDA_VISIBLE_DEVICES=<gpu-id> python  unsupervised.py  --src_lang <source-language> --tgt_lang <target-language> --src_emb <source-embedding-path> --tgt_emb <target-embedding-path> --max_vocab_A <max-vocab-size-source> --max_vocab_B <max-vocab-size-target> --dis_most_frequent_AB <most-freq-emb-src2tgt-adversary> --dis_most_frequent_BA <most-freq-emb-tgt2src-adversary> --normalize_embeddings <normalizing-values> --emb_dim_autoenc <code-space-dimension> --dis_lambda <adversarial-loss-weight> --cycle_lambda <cycle-loss-weight> --reconstruction_lambda <reconstruction-loss-weight>

You can also control other hyperparameters.
For example to run EN-ES:

```bash
CUDA_VISIBLE_DEVICES=0 python  unsupervised.py  --src_lang en --tgt_lang es --src_emb ./data/wiki.en.vec --tgt_emb ./data/wiki.es.vec --max_vocab_A 200000 --max_vocab_B 200000 --dis_most_frequent_AB 50000 --dis_most_frequent_BA 50000  --normalize_embeddings 'renorm,center,renorm' --emb_dim_autoenc 350 --dis_lambda 1 --cycle_lambda 5 --reconstruction_lambda 1 
```
You will get the word translation accuracies at different precision (1, 5, 10) for EN-ES and ES-EN.


## References
Please cite our paper if you found the resources in this repository useful.
```bash
@InProceedings{mohiuddin-joty-naacl-19,
     title="{Revisiting Adversarial Autoencoder for Unsupervised Word Translation with Cycle Consistency and Improved Training}",
     author={Tasnim Mohiuddin and Shafiq Joty},
     booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
     series    ={NAACL-HLT'19},
     publisher={Association for Computational Linguistics},
     address   = {Minneapolis, USA},
     pages={xx--xx},
     url = {},
     year={2019}
}

```

