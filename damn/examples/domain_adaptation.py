import torch
from torchvision import transforms, datasets
import argparse
from torch import nn

import pytorch_lightning as pl

from damn import DAMNTrain,DomainDataset 


class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y =self.dataset[idx]
        return x, -1


class DigitDataModule(pl.LightningDataModule):
    def __init__(self, labeled_data, unlabeled_data, test_data, batch_size):
        super().__init__()
        transform = transforms.Compose([
                        transforms.Grayscale(),
                        transforms.Resize(32),
                        transforms.ToTensor()
                        ])
        self.dataset_by_name = {"MNIST" : lambda train: datasets.MNIST("./data/mnist", train=train, transform=transform,download=True),
                                "USPS": lambda train: datasets.USPS("./data/usps", train=train, transform=transform,download=True),
                                "SVHN": lambda train: datasets.SVHN("./data/svhn", transform=transform,download=True, split="train" if train else "test")}
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.test_data = test_data
        self.batch_size = batch_size
        

    def setup(self, stage=None):
        supervised = [DomainDataset(self.dataset_by_name[ds](train=True), idx) for idx, ds in enumerate(self.labeled_data)]
        unsupervised = [DomainDataset(UnsupervisedDataset(self.dataset_by_name[ds](train=True)), idx) for idx, ds in enumerate(self.unlabeled_data)]
        dataset = supervised + unsupervised
        self.train_dataset = torch.utils.data.ConcatDataset(dataset)
        self.test_dataset = self.dataset_by_name[self.test_data](train=False)


    def train_dataloader(self):
        return  torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

class DAMNLeNet(DAMNTrain):
    def __init__(self, num_domain, lr=0.01, domain_coef=0.1, consistency_coef=0.1):
        backbone = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
            torch.nn.Flatten()
        )
        classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )
        DAMNTrain.__init__(self,backbone, classifier, 120,num_domain,domain_coef,consistency_coef)
        self.cls_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.accuracy = pl.metrics.Accuracy()
        self.lr = lr


    def forward(self, x):
        logits = super(DAMNLeNet, self).forward(x)
        return torch.nn.functional.softmax(logits, dim=1)
    

    def training_step(self, batch, batch_idx):
        (x, y), domain = batch
        #Determine samples without label
        cls_weight = 1.0 - (y == -1).float()
        y=torch.abs(y)
        auxiliary_loss, logits = super(DAMNLeNet, self).training_step((x,domain), batch_idx)
        classification_loss = torch.mean(cls_weight * self.cls_loss(logits, y))
        self.log('train_accuracy', self.accuracy(logits, y), prog_bar=True, on_step=True)
        return classification_loss + auxiliary_loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        self.log('test_accuracy', self.accuracy(pred, y),prog_bar=True, on_step=True)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--domain_coef', type=float, default=0.1)
        parser.add_argument('--consistency_coef', type=float, default=0.1)
        return parser


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--supervised-datasets', nargs='+', default=["MNIST"], help='Training datasets with labels. Possible choices:[MNIST,USPS,SVHN]')
    parser.add_argument('--unsupervised-datasets', nargs='+', default=[], help='Training datasets without labels. Possible choices:[MNIST,USPS,SVHN]')
    parser.add_argument('--test-datasets', default="MNIST", choices=["MNIST","USPS","SVHN"], help='Select testset')
    parser.add_argument('--batch-size', default=32, help='Batch size')
    parser.add_argument('--epoch', default=30, help='Epoch')
    parser = DAMNLeNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = DAMNLeNet(len(args.supervised_datasets) + len(args.unsupervised_datasets),args.lr, args.domain_coef, args.consistency_coef)
    digit_data = DigitDataModule(args.supervised_datasets,args.unsupervised_datasets, args.test_datasets, args.batch_size)

    trainer = pl.Trainer(max_epochs=args.epoch)
    trainer.fit(model, datamodule=digit_data)
    trainer.test(model, datamodule=digit_data)


    

    