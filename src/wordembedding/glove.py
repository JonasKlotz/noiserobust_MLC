from __future__ import annotations

import shutil
from pathlib import Path

import urllib3
from scipy import spatial
# from simple_downloader import download
from collections import defaultdict
import zipfile
import numpy as np
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE  # this can reduce our big dataset into lesser dimensions

GLOVE_URL = "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"


class Glove:
    """
    This class loads and processes the weights for a Glove word2vec embedding
    """

    def __init__(self, data_path="data/glove", load_from_txt=False, download=False, padding_vector=None,
                 unknown_vector=None, dim=50):
        """
        Initialize Glove weights from a given path.
        :param download: Initializing with download can take quite some time, as 800 MB have to be downloaded.
        :param data_path: path to a directory where the weights will be saved. I

        """
        self.dim = dim
        self.data_path = data_path
        if download:
            self.words, self.vectors = self.download_glove()
            print(f"loaded GLOVE in {self.data_path}")
        elif load_from_txt:
            self.words, self.vectors = self._load_from_text()
            print(f"loaded GLOVE from txt file {self.data_path}")
        else:
            self.words, self.vectors = self.load_glove()
            print(f"loaded GLOVE from {self.data_path}")

        self._set_padding_and_unknown(padding_vector=padding_vector, unknown_vector=unknown_vector)
        self._setup_dict()
        self.save_glove()
        self._clean_directory()

    def __getitem__(self, item: list | str) -> list | np.ndarray:
        """
        overrides such that the Glove object can be accesses like
        **Example**::
            g = Glove(data_path)
            the_vector = g["the"]
        :param item: list or str type
        :return: np.array or list of np.arrasys
        """
        if isinstance(item, str):
            return self.dictionary[item.lower()]
        if isinstance(item, list):
            return [self.dictionary[i.lower()] for i in item]

    def download_glove(self) -> (np.ndarray, np.ndarray):
        """
        Downloads glove unpacks it and processes it into np.arrays
        :return: words and their glove vector representation
        """
        download_dir = Path(self.data_path)
        download_dir.mkdir(exist_ok=True)

        output_file = "glove.zip"
        http = urllib3.PoolManager()
        with open(output_file, 'wb') as out:
            r = http.request('GET', GLOVE_URL, preload_content=False)
            shutil.copyfileobj(r, out)

        zipf = zipfile.ZipFile(output_file)
        zipf.extractall(path=self.data_path)

        return self._load_from_text()

    def _load_from_text(self):
        vocab, embeddings = [], []
        with open(f"{self.data_path}/glove.6B.{str(self.dim)}d.txt", 'rt') as fi:
            full_content = fi.read().strip().split('\n')
        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocab.append(i_word)
            embeddings.append(i_embeddings)
        return np.array(vocab), np.array(embeddings)

    def _set_padding_and_unknown(self, padding_vector=None, unknown_vector=None):
        """
        This method resets the vector representation of the padding and for unknown words.
        Todo: very inefficient
        :param pad_emb_npa:
        :param unk_emb_npa:
        :return:
        """
        if not padding_vector:
            self.padding_vector = np.zeros((1, self.vectors.shape[1]))  # embedding for '<pad>' token.

        if not unknown_vector:
            self.unknown_vector = np.mean(self.vectors, axis=0, keepdims=True)  # embedding for '<unk>' token.

        self.vectors = np.vstack((self.padding_vector, self.unknown_vector, self.vectors))

    def _setup_dict(self):
        # setup a dictionary with a default value
        self.dictionary = defaultdict(lambda: self.unknown_vector)
        # for all words add the corresponding vector
        for i, w in enumerate(self.words):
            self.dictionary[w] = self.vectors[i]

    def save_glove(self):
        """
        saves glowe as npy files in the data dir
        :return:
        """
        with open(f'{self.data_path}/words_{str(self.dim)}.npy', 'wb') as f:
            np.save(f, self.words)

        with open(f'{self.data_path}/vectors_{str(self.dim)}.npy', 'wb') as f:
            np.save(f, self.vectors)

    def load_glove(self):
        """
        loads glove embedding from NPY files contained in the data dir
        :return:
        """
        try:
            words = np.load(f'{self.data_path}/words_{str(self.dim)}.npy')
            vectors = np.load(f'{self.data_path}/vectors_{str(self.dim)}.npy')
        except OSError as e:
            print(e)
            raise FileNotFoundError(f"No files at {self.data_path}")
        return words, vectors

    def _clean_directory(self):
        """
        Remove all files that are not the NPY weight files from the data_dir

        :return:
        """
        cwd = os.getcwd()
        fileList = glob.glob(cwd + self.data_path + "/*")
        for file in fileList:
            if "words" in file or "vectors" in file:
                continue
            else:
                try:
                    os.remove(file)
                except OSError as e:
                    print("Error: %s : %s" % (file, e.strerror))

    def find_closest_embeddings(self, embedding):
        """
        Given a embedding vector this returns the most close word contained in glove.

        This can be used to retrieve a "name" for new word for instance created by adding up two old words.
        :return:
        """
        return sorted(self.dictionary.keys(),
                      key=lambda word: spatial.distance.euclidean(self.dictionary[word], embedding))

    def plot_words(self, words):
        """
        Visualizes the distance between the given words
        :param words: list of words to be plotted
        :return:
        """
        tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
        n, d = self.vectors.shape  # number of samples, dimensionality

        vectors = np.asarray([self.dictionary[word] for word in words])

        # T-distributed Stochastic Neighbor Embedding.
        # this learns how to optimally represent the vectors in a 2d space

        Y = tsne.fit_transform(vectors)

        plt.scatter(Y[:, 0], Y[:, 1])

        for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
        plt.show()

    def save_word_embeddings(self, words, name="deepglobe"):
        res = self[words]
        with open(f'{self.data_path}/{name}_labels_d{str(self.dim)}.npy', 'wb') as f:
            np.save(f, res)

    def reduce_vector_dimension(self, d=10):
        tsne = TSNE(n_components=d, random_state=0, init='pca', learning_rate='auto', method='exact' )
        self.vectors = tsne.fit_transform(self.vectors)


import networkx as nx


def plot_edge_matrix(edge_matrix, label_mapping):
    G = nx.from_numpy_matrix(edge_matrix)
    G = nx.relabel_nodes(G, label_mapping)

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 7]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 7]

    pos = nx.circular_layout(G)  # positions for all nodes - seed for reproducibility
    # pos = nx.kamada_kawai_layout(G)  # positions for all nodes - seed for reproducibility
    fig = plt.figure(figsize=(10, 10))
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=4000)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=4)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=15, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    fig.savefig('waka.svg')


def load_word_embeddings(dim, labels, data_path="data/glove", name="deepglobe"):
    try:
        return np.load(f'{data_path}/{name}_labels_d{str(dim)}.npy')
    except:
        G_dict = Glove(dim=dim)
        return G_dict[labels]


if __name__ == '__main__':
    G_dict = Glove(dim=50, load_from_txt=True)  # Download = True can can take long
    # print(G_dict["nichtdrinne"])

    G_dict.reduce_vector_dimension(d=15) # reduce embedded space of the vectors

    labels = ["urban", "agriculture", "rangeland", "forest", "water", "barren", "unknown"]
    G_dict.save_word_embeddings(words=labels)
    # label_mapping = {idx: l for idx, l in enumerate(labels)}
    n = len(labels)
    res = (G_dict[labels])
    # edge_matrix = np.zeros((n, n))
    #
    # for i in range(n):
    #     for k in range(n):
    #         edge_matrix[i, k] = np.linalg.norm(res[i] - res[k])
    #
    # edge_matrix = np.round(edge_matrix, 2)
    # plot_edge_matrix(edge_matrix, label_mapping)
    # """
    #     G = nx.from_numpy_matrix(edge_matrix)
    #     # Create a non-interactive plot:
    #     Graph(G)
    #     plt.show()"""
