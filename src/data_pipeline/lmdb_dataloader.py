from torch.utils.data import Dataset, DataLoader
import lmdb
import os



class LMDBLoader(Dataset):

    def __init__(self, path, transform, translation_dict):
        self.path = path
        self.env = None
        self.keys = None
        self.len = None

    def _init_db(self):
        self.env = lmdb.open(self.path, subdir=os.path.isdir(self.path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin()
        # get all keys and the overall length of the dataset
        self.len = self.txn.stat()['entries']
        self.keys = [key for key, _ in self.txn.cursor()]

    def __len__(self):
        if self.len is None:
            self._init_db()
        return self.len

    def __getitem__(self, idx):
        # Delay loading LMDB data until after initialization
        if self.env is None:
            self._init_db()

        key = self.keys[idx]
        value = self.txn.get(key, readonly=True, buffers=True)

        img = value['img']
        labels = value['labels']

        return {'image': img, 'labels': labels}


def load_data():
    pass
