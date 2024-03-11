# main.py
import os  
import torch  
import torch.multiprocessing as mp  
from torch import nn  
from torch.distributed import init_process_group, destroy_process_group  
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.utils.data import DataLoader, Dataset  
from torch.utils.data.distributed import DistributedSampler  
from torchvision import datasets  
from torchvision.transforms import ToTensor  

MAX_EPOCHS = 10  
SAVE_EVERY = 1  
if torch.cuda.is_available():
    device = torch.device('cuda') #GPU이용
    
else:
    device = torch.device('cpu') #GPU이용안되면 CPU이용
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

class Trainer:  
    def __init__(  
        self,  
        model: torch.nn.Module,  
        train_data: DataLoader,  
        optimizer: torch.optim.Optimizer,  
    ) -> None:  
        self.model = DDP(model.to(device))  
        self.train_data = train_data  
        self.optimizer = optimizer  

    def _run_epoch(self, epoch):  
        b_sz = len(next(iter(self.train_data))[0])  
        print(f"b_sz : {b_sz} / epoch : {epoch} / len data : {len(self.train_data)}")  
        for source, targets in self.train_data:  
            source = source.to(device)  
            targets = targets.to(device)  
            self.optimizer.zero_grad()  
            output = self.model(source)  
            loss = torch.nn.CrossEntropyLoss()(output, targets)  
            loss.backward()  
            self.optimizer.step()  

    def _save_checkpoint(self, epoch):  
        ckp = self.model.module.state_dict()  
        torch.save(ckp, "ckpt.pt")  
        print(f"Epoch {epoch} | Training ckpt saved at ckpt.pt")  

    def train(self):  
        for epoch in range(MAX_EPOCHS):  
            self._run_epoch(epoch)  
            if epoch % SAVE_EVERY == 0:  
                self._save_checkpoint(epoch)  


class NeuralNetwork(nn.Module):  
    def __init__(self):  
        super().__init__()  
        self.flatten = nn.Flatten()  
        self.linear_relu_stack = nn.Sequential(  
            nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)  
        )  

    def forward(self, x):  
        x = self.flatten(x)  
        logits = self.linear_relu_stack(x)  
        return logits  


def load_train_dataset_model_and_opt():  
    train_set = datasets.FashionMNIST(  
        root="data",  
        train=True,  
        download=True,  
        transform=ToTensor(),  
    )  

    model = NeuralNetwork()  
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  
    return train_set, model, optimizer  


def prepare_dataloader(dataset: Dataset, batch_size: int):  
    return DataLoader(  
        dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset)  
    )  


def main(rank: int): # rank 는 mp.spawn에서 자동으로 할당하기 때문에 사용하지 않아도 받아야 한다. 그렇지 않으면 에러가 난다.  
    init_process_group(backend="nccl")
    dataset, model, optimizer = load_train_dataset_model_and_opt()  
    print(f"WORLD_SIZE : {os.environ['WORLD_SIZE']} / len(dataset) : {len(dataset)}") # pytorchjob 이 분산처리에 필요한 인자들을 os.environ 에 자동으로 할당한다  
    train_data = prepare_dataloader(dataset, batch_size=32)  
    trainer = Trainer(model, train_data, optimizer) # 기존과 달리 모델을 특정 gpu_id에 올려줄 필요가 없다.  
    trainer.train()  
    destroy_process_group()  


if __name__ == "__main__":  
    torch.cuda.empty_cache()  
    mp.spawn(  
        main,
    )
