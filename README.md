# Evaluating Multi-label Classifiers with Noisy Labels
https://arxiv.org/pdf/2102.08427.pdf

# Run Example

python main.py -dataset cars -batch_size 32 -d_model 300 -d_inner_hid 512 -n_layers_enc 2 -n_layers_dec 2 -n_head 4 -epoch 5 -dropout 0.2 -dec_dropout 0.2 -lr 0.0002 -encoder resnet -decoder graph -label_mask prior

# Data Set (Deep Globe Patches)

Die Data-Pipeline für DeepGlobe ist unter `data_pipeline/deepglobe/patch_sampling.py` ausführbar:
Zuvor muss nur das Deepglobe Datenset heruntergeladen werden und in das Verzeichnis `data/deepglobe` entpackt werden.
Die Patches der Bilder werden mit ihren korrespondierenden Labels zusammen im LMDB-Format gespeichert:

`data/deepglobe_patches/[train/test/valid]/`

Das Format des Value Strings ist hierbei für jeden Eintrag immer:

`key: [sample_id]_[patch_index]` 

`value: dict{'x': [x-coordinate], 'y': [y-coordinate], 'img': [subsample image],'label': [label names present]}`



# Helpful TU Links
• Kai notebook https://docs.kai-tub.tech/ben-docs/00_intro.html

• Kai repo https://github.com/kai-tub/ben-docs

• LaMP: https://github.com/QData/LaMP

• Updated LaMP (with Transformers): https://github.com/QData/C-Tran

# Paper References
• [1] https://www.image-net.org/, https://paperswithcode.com/sota/image-classification-on-imagenet

• [2] https://bigearth.net/, https://arxiv.org/abs/2105.07921

• [3] https://arxiv.org/abs/2102.08427

• [4] https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset

• [5] https://arxiv.org/abs/1904.03582, https://github.com/Megvii-Nanjing/ML-GCN

• [6] https://arxiv.org/abs/2009.14119, https://github.com/Alibaba-MIIL/ASL

• [7] Lamp: https://arxiv.org/pdf/1904.08049.pdf


# TODO

Da wir alle in diesem Repo arbeiten werden, brauchen wir extrem eindeutige Namensgebung die sofort intuitiv klar sind.

Bitte lieber alles sehr lang benennen als zu kurz!

Falls etwas im Repo nichtmehr direkt gebraucht wird -> `archive`

# Dependencies
Python version:  Python 3.7.1

