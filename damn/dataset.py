from torch.utils.data import Dataset

class DomainDataset(Dataset):
    def __init__(self, dataset, domain_idx):
        self.dataset = dataset
        self.domain_idx = domain_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.domain_idx