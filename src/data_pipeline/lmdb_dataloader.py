from torch.utils.data import Dataset, DataLoader



class LMDBLoader(Dataset):

    def __init__(self, path, transform, translation_dict):
        self.path = path


    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):


        sample = {'image': "single_img",
                  'labels': "label_tensor"}
        return sample


def load_data