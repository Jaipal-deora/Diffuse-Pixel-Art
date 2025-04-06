import numpy as np
import torch 
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PixelSet(Dataset):
    def __init__(self,sprites_path,labels_path,transform,null_context=False):
        self.sprites = np.load(sprites_path)
        self.labels = np.load(labels_path)
        self.transform = transform 
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape 
        self.labels_shape = self.labels.shape 

    def __len__(self):
        return self.sprites_shape[0]

    def __getitem__(self,idx):
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.labels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        return self.sprites_shape, self.labels_shape
    

def get_dataloader(sprites_path,labels_path,batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,))  # range [-1,1]

    ])

    dataset = PixelSet(sprites_path,labels_path, transform, null_context=False)
    return DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1)


