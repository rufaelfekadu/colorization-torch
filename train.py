import torch
from torch import nn
from model import ColorizeNet, ColorizeLoss
from data import CustomDataset, collate_fn

class SimpleTrainer:
    def __init__(self, model, optimizer, criterion, train_loader, device):
        self.batch_size = 32
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        # self.val_loader = val_loader
        # self.test_loader = test_loader
        self.device = device

    def train(self, num_epochs):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for data in self.train_loader:
                data_l, gt_ab_313, prior_boost_nongray = data
                data_l, gt_ab_313, prior_boost_nongray = data_l.to(self.device), gt_ab_313.to(self.device), prior_boost_nongray.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data_l)
                loss = self.criterion(outputs, prior_boost_nongray, gt_ab_313)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(self.train_loader)}")
            # save checkpoint
            self.save(f'checkpoint.pth')

    # def val(self):
    #     self.model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for data in self.val_loader:
    #             data_l, gt_ab_313, prior_boost_nongray = data
    #             data_l, gt_ab_313, prior_boost_nongray = data_l.to(self.device), gt_ab_313.to(self.device), prior_boost_nongray.to(self.device)
    #             outputs = self.model(data_l)
    #             loss = self.criterion(outputs, prior_boost_nongray, gt_ab_313)
    #             val_loss += loss.item()
    #     print(f"Validation Loss: {val_loss/len(self.val_loader)}")

    def test(self):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                data_l, gt_ab_313, prior_boost_nongray = data
                data_l, gt_ab_313, prior_boost_nongray = data_l.to(self.device), gt_ab_313.to(self.device), prior_boost_nongray.to(self.device)
                outputs = self.model(data_l)
                loss = self.criterion(outputs, prior_boost_nongray, gt_ab_313)
                test_loss += loss.item()
        print(f"Test Loss: {test_loss/len(self.test_loader)}")

    def save(self, path='model.pth'):
        to_save = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(to_save, path)
    
    def load(self, path='model.pth'):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    # build model
    model = ColorizeNet()
    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # build loss
    criterion = ColorizeLoss(batch_size=32)
    # build data loaders
    common_params = {'image_size': 224, 'batch_size': 32}
    dataset_params = {'path': 'path/to/file.txt'}
    train_set = CustomDataset(common_params=common_params, dataset_params=dataset_params)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    trainer = SimpleTrainer(model, optimizer, criterion, train_loader, device='cuda')
    trainer.train(num_epochs=10)

if __name__ == '__main__':
    main()


