import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(*[nn.Residual(nn.Sequential(*[
                                            nn.Linear(dim, hidden_dim),
                                            norm(hidden_dim),
                                            nn.ReLU(),
                                            nn.Dropout(drop_prob),
                                            nn.Linear(hidden_dim, dim),
                                            norm(dim)
                                       ])), 
                           nn.ReLU()
           ])
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    blocks = []
    blocks.append(nn.Linear(dim, hidden_dim))
    blocks.append(nn.ReLU())
    for _ in range(num_blocks):
        blocks.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    blocks.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*blocks)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()

    err_cnt = 0
    losses = []
    for data, label in dataloader:
        logits = model(data.reshape((-1, 28*28)))
        err_cnt += np.sum(np.argmax(logits.numpy(), axis=1) != label.numpy())
        loss = nn.SoftmaxLoss()(logits, label)

        if opt:
            loss.backward()
            opt.step()
            opt.reset_grad()
        
        losses.append(loss.numpy())
    return err_cnt/len(dataloader.dataset), np.mean(losses)
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model = MLPResNet(dim=28*28, hidden_dim=hidden_dim)
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'),
                                          os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
                                         )
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, True)

    test_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'),
                                          os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
                                         )
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size, True)

    for _ in range(epochs):
        train_err, train_loss = epoch(train_dataloader.__iter__(), model, optimizer)
    
    test_err, test_loss = epoch(test_dataloader.__iter__(), model)
    return 1-train_err, train_loss, 1-test_err, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
