from models import Model
from natthaphon import Progbar
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import DensoDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch


if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomResizedCrop(224),
                                          # transforms.RandomHorizontalFlip(),
                                          # transforms.RandomVerticalFlip(),
                                          transforms.ColorJitter(0.2, 0.2, 0.2, 0.4),
                                          transforms.ToTensor(),
                                          normalize,
                                          ])

    dataset = DensoDataset(transform=train_transform)
    indices = list(range(len(dataset)))
    split = int(np.floor(0.3 * len(dataset)))
    np.random.seed(8888)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=4,
                              sampler=train_sampler, num_workers=2)
    validation_loader = DataLoader(dataset, batch_size=4,
                                   sampler=valid_sampler, num_workers=2)

    model = Model()
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(30):
        print('Epoch', epoch + 1)
        progbar = Progbar(len(train_loader))
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            acc = predicted.eq(targets.long()).double().sum() / targets.size(0)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            printlog = [['loss', loss.cpu().detach().numpy()], ['acc', acc.cpu().detach().numpy()]]
            progbar.update(idx + 1, printlog)
        model.eval()
        progbar = Progbar(len(validation_loader))
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(validation_loader):
                inputs = inputs.cuda()
                targets = targets.cuda()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_acc = predicted.eq(targets.long()).double().sum() / targets.size(0)
                val_loss = criterion(outputs, targets)
                printlog = [['val_loss', val_loss.cpu().detach().numpy()], ['val_acc', val_acc.cpu().detach().numpy()]]
                progbar.update(idx + 1, printlog)
        torch.save(model.state_dict(), f'/media/palm/BiggerData/denso/checkpoints/{epoch}')
