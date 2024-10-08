import torch
from torch import nn
from model import ColorizeNet, ColorizeLoss, eccv16
from data import build_dataloader
from tqdm import tqdm
from PIL import Image
import numpy as np
from skimage.io import imsave


class SimpleTrainer:
    def __init__(self, model, optimizer, criterion, train_loader, test_loader, device):
        self.batch_size = 32
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        # self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.epoch = 0
        self.global_step = 0
        self.n_steps_to_plot = 10

    def train(self, num_epochs):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            tq_iter = tqdm(self.train_loader, leave=False)
            for data in tq_iter:
                data_l, gt_ab_313, prior_boost_nongray = data
                data_l, gt_ab_313, prior_boost_nongray = data_l.to(self.device), gt_ab_313.to(self.device), prior_boost_nongray.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data_l)
                loss, _ = self.criterion(outputs, prior_boost_nongray, gt_ab_313)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if self.global_step % self.n_steps_to_plot == 0:
                    tq_iter.set_description(f"Loss: {running_loss/len(self.train_loader)}")
                    self.test()
                self.global_step += 1
            # save checkpoint
            self.epoch += 1
            self.save(f'outputs/checkpoint.pth')

    def test(self):
        self.model.eval()
        samples_dir = 'outputs/predictions'
        gt_dir = 'outputs/gt'
        with torch.no_grad():
            data =  next(iter(self.test_loader))
            data_l, data_rgb = data
            data_l = data_l.to(self.device)
            outputs = self.model(data_l)
            # save image
            for i in range(outputs.shape[0]):
                out_rgb = self.model.decode(data_l[i].unsqueeze(0).cpu().numpy(), outputs[i].unsqueeze(0).cpu().numpy(), rebalance=2.63)
                out_rgb = (out_rgb*255).astype(np.uint8)
                imsave(f'{samples_dir}/{i}.png', out_rgb)
                gt_img = (data_rgb[i]*255).numpy().astype(np.uint8)     
                imsave(f'{gt_dir}/{i}.png', gt_img)
            

        self.model.train()

    def save(self, path='model.pth'):
        to_save = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
        }
        torch.save(to_save, path)
    
    def load(self, path='model.pth'):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    # build model
    model = eccv16(pretrained=True)
    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # build loss
    criterion = ColorizeLoss(batch_size=32)
    # build data loaders
    train_paths = 'resources/train.txt'
    test_paths = 'resources/train.txt'
    train_loader = build_dataloader(train_paths, batch_size=32, num_workers=4, split='train')
    test_loader = build_dataloader(test_paths, batch_size=32, num_workers=4, split='test')
    
    trainer = SimpleTrainer(model, optimizer, criterion, train_loader, test_loader, device='cuda')
    trainer.train(num_epochs=10)

if __name__ == '__main__':
    main()


