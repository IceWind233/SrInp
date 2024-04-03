# train softmax splatting network

# first, load DITN model and make first layer as the input of softmax splatting network
# second, load softmax splatting network
# third, ground truth is the middle frame of the input sequence
# fourth, train the softmax splatting network

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models.feature_extraction import create_feature_extractor

import numpy as np
from PIL import Image
import json

from model import DITN_Real, SoftmaxSplatting

def feature_extractor(DITN_model):
    node = 'sft'
    return create_feature_extractor(DITN_model, return_nodes=[node])

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def to_tensor(data_paths):
    sequences = []

    for item in data_paths:
        images = []

        for path in item["images"]:
            image = Image.open(path)
            image = np.array(image).transpose(2, 0, 1)
            images.append(image)
        sequences.append(images)
        
    sequences = np.array(sequences)
    return sequences

def load_dataset(path, batch_size):
    data_path = load_json(path)
    dataset = to_tensor(data_path)
    
    return split_dataset(DataLoader(dataset, batch_size=batch_size, shuffle=True))

def split_dataset(dataloader):
    train_size = int(0.8 * len(dataloader))
    test_size = len(dataloader) - train_size
    train_dataset, test_dataset = random_split(dataloader, [train_size, test_size])
    return train_dataset, test_dataset

# load DITN model and make first layer as the input of softmax splatting network
# the inputs of softmax splatting network are the output of the first layer of DITN model of the first frame and the third frame
# the ground truth is the output of the first layer of DITN model of the second frame
def train(softmax_path, ditn_path, data_path, batch_size, epochs):
    # create "trained" dir at models
    os.makedirs(os.path.join("model", "trained"), exist_ok=True)

    # load dataset
    train_dataset, test_dataset = load_dataset(data_path, batch_size)

    # create model
    # load DITN model
    ditn_model = DITN_Real()
    ditn_model.load_state_dict(torch.load(ditn_path))
    
    ditn_model.eval()
    # load softmax splatting network
    softmax = SoftmaxSplatting()
    softmax.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(softmax_path).items()})
    # softmax.load_state_dict(torch.load(softmax_path))
    if torch.cuda.is_available():
        ditn_model.cuda()
        softmax.cuda()

    # optimizer and scheduler
    optimizer = optim.Adam(softmax.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50, 70, 90], gamma=0.1)

    # loss function
    criterion = nn.MSELoss()

    # tensorboard
    writer = SummaryWriter()

    extractor = feature_extractor(ditn_model)

    # train
    for epoch in range(epochs):
        softmax.train()
        for _, images in enumerate(train_dataset.dataset):
            # print(np.array(images).shape)
            for i in range(len(images[0]) - 2):
                tensors = images[0][i:i + 3]

                tensor1 = torch.FloatTensor(np.ascontiguousarray([tensors[0]]))
                tensor2 = torch.FloatTensor(np.ascontiguousarray([tensors[1]]))
                tensor3 = torch.FloatTensor(np.ascontiguousarray([tensors[2]]))

                tensor1 = tensor1.cuda()
                tensor2 = tensor2.cuda()
                tensor3 = tensor3.cuda()

                # transform to float tensor and normalize
                tensor1 = tensor1.float() / 255.0
                tensor2 = tensor2.float() / 255.0
                tensor3 = tensor3.float() / 255.0

                # forward
                sftTensor1 = extractor(tensor1)['sft']
                sftTensor2 = extractor(tensor2)['sft']
                sftTensor3 = extractor(tensor3)['sft']
                print("\n===>", sftTensor1.cpu().shape, "\n")
                # forward
                # the inputs of softmax splatting network are the output of the first layer of DITN model of the first frame and the third frame
                # the ground truth is the output of the first layer of DITN model of the second frame
                output = softmax(sftTensor1, sftTensor3, [0.5])
                loss = criterion(output, sftTensor2)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # log
                print(f'epoch: {epoch}, iter: {i}, loss: {loss.item()}')
                writer.add_scalar('train/loss', loss.item(), epoch * len(train_dataset) + i)
        # eval
        softmax.eval()
        with torch.no_grad():
            for _, images in enumerate(test_dataset.dataset):
                for i in range(len(images) - 2):
                    tensors = images[0][i:i + 3]

                    tensor1 = torch.FloatTensor(np.ascontiguousarray(tensors[0]))
                    tensor2 = torch.FloatTensor(np.ascontiguousarray(tensors[1]))
                    tensor3 = torch.FloatTensor(np.ascontiguousarray(tensors[2]))

                    tensor1 = tensor1.cuda()
                    tensor2 = tensor2.cuda()
                    tensor3 = tensor3.cuda()

                    tensor1 = tensor1.float() / 255.0
                    tensor2 = tensor2.float() / 255.0
                    tensor3 = tensor3.float() / 255.0

                    # forward
                    sftTensor1 = extractor(tensor1)['sft']
                    sftTensor2 = extractor(tensor2)['sft']
                    sftTensor3 = extractor(tensor3)['sft']
                    

                    # forward
                    output = softmax(sftTensor1, sftTensor3, [0.5])
                    loss = criterion(output, sftTensor2)

                    # log
                    print(f'epoch: {epoch}, iter: {i}, loss: {loss.item()}')
                    writer.add_scalar('val/loss', loss.item(), epoch * len(test_dataset) + i)
        scheduler.step()
        # save model
        if epoch % 9 == 0:
            torch.save(softmax.state_dict(), f'models/trained/softmaxsplatting_epoch_{epoch}.pth')

    writer.close()
        

if __name__ == '__main__':
    import argparse
    # read argumentsimport argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--softmax_path', type=str, default="models/network-lf.pytorch", help="path to the softmax splatting network")
    parser.add_argument('--ditn_path', type=str, default="models/DITN_Real_x4.pth", help="path to the DITN model")
    parser.add_argument('--data_path', type=str, default="metadata.json", help="path to the metadata.json which contains the paths of the input sequences")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # train
    train(
        softmax_path=args.softmax_path,
        ditn_path=args.ditn_path,
        data_path=args.data_path,
        batch_size=args.batch_size,    
        epochs=args.epochs)
    
    print("done!!")

