import sns as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import confusion_matrix

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transforms.Normalize((mean_R, mean_G, mean_B), (std_R, std_G, std_B)) bu yüzden 0.5 oluyor. Bu dönüşüm sayesinde
# veriler ölçeklendirilebiliyor ve daha iyi optimizasyon sağlanıyor

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


batch_boyut = 32 # kaç tane batch seçileck internette baktığımda genelde hep iki katları seçilmiş 32 64 128 256 gibi ben 32 seçtim
trainSet = DataLoader(train, batch_size=batch_boyut, shuffle=True)
Testset = DataLoader(test, batch_size=batch_boyut, shuffle=False)

tenserboard = SummaryWriter()# Tenserboard için


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


# Deney parametreleri
epochs = 100##100 adım

LR = [1e-2,
      1e-6]

out = 10  ## sınıf sayısı

layer = 10 ## Ben ayarladım

inp = 3 * 32 * 32  ## cifarın görüntü boyutları


# Optimizer  algoritmaları
op = {
    'RMSprop': optim.RMSprop,
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'Adagrad': optim.Adagrad,
}


for op_name, op_sınıf in op.items():
    for lr in LR:
        a = MLP(inp, layer, out).to(device)
        optimizer = op_sınıf(a.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        x = f'Optmzr-{op_name}_LR_{lr}'
        acc_train, loss_train = [], []
        acc_val, loss_val = [], []

        for epoch in range(epochs):
            m=0
            n=0
            p = 0.0
            a.train()


            for i, j in trainSet:
                i =i.to(device)
                j = j.to(device)

                optimizer.zero_grad()

                outputs = a(i)
                loss = criterion(outputs, j)
                loss.backward()
                optimizer.step()

                p += loss.item()
                _, predicted = outputs.max(1)
                n += j.size(0)
                m += predicted.eq(j).sum().item()

            train_acc = m / n
            train_loss = p / len(trainSet)
            # TensorBoard'a kayıt

            tenserboard.add_scalar(f'{x}/Train_Acc', train_acc, epoch)
            tenserboard.add_scalar(f'{x}/Train_Loss', train_loss, epoch)

            a.eval()
            k = 0.0
            l = 0
            t = 0

            with torch.no_grad():
                for i, j in Testset:
                    i, j = i.to(device), j.to(device)

                    outputs = a(i)
                    loss = criterion(outputs, j)

                    k += loss.item()
                    _, predicted = outputs.max(1)
                    t += j.size(0)
                    l += predicted.eq(j).sum().item()
            val_acc = l / t
            val_loss = k / len(Testset)

            #  her bir adım sonuçları
            acc_train.append(train_acc)
            loss_train.append(train_loss)
            acc_val.append(val_acc)
            loss_val.append(val_loss)

            tenserboard.add_scalar(f'{x}/Val_Acc', val_acc, epoch)
            tenserboard.add_scalar(f'{x}/Val_Loss', val_loss, epoch)
            print(f'Epoch   -    {epoch + 1}  - Train_Acc: {train_acc:.4f}  -   Val_Acc: {val_acc:.4f}  -  '
                  f' Train_Loss: {train_loss:.4f}-   Val_Loss: {val_loss:.4f}')

        # TensorBoard'a grafikleri kaydet
        tenserboard.add_figure(f'{x}/Accuracy', plt.figure())
        tenserboard.add_figure(f'{x}/Loss', plt.figure())

        tenserboard.close()

        ## Sonuçların grafikle gösterip png olarak kayıt eder. Rapordaki görüntüleri bu şekilde elde ettim

        
        plt.plot(range(epochs), loss_val, label='Validation Loss')
        plt.plot(range(epochs), acc_val, label='Validation Accuracy')
        plt.xlabel('Adım')
        plt.ylabel('Başarı')
        plt.legend()


        plt.savefig(f'1ValidationAccuracyLost{op_name}-{LR}.png')
        plt.close()

        plt.plot(range(epochs), loss_train, label='Train Loss')
        plt.plot(range(epochs), acc_train, label='Train Accuracy')
        plt.xlabel('Adım')
        plt.ylabel('Kayıp')
        plt.legend()
        plt.savefig(f'2TrainAccurLoss{op_name}-{LR}.png')
        plt.close()
        plt.plot(range(epochs), loss_train, label='Train Loss')
        plt.plot(range(epochs), val_loss, label='Val Loss')
        plt.xlabel('Adım')
        plt.ylabel('Kayıp')
        plt.legend()
        plt.savefig(f'3TrainValLoss{op_name}-{LR}.png')
        plt.close()
        plt.plot(range(epochs), acc_train, label='Train Acc')
        plt.plot(range(epochs), acc_val, label='Val Acc')
        plt.xlabel('Adım')
        plt.ylabel('Başarı')
        plt.legend()
        plt.savefig(f'3TrainValAcc{op_name}-{LR}.png')
        plt.close()

        # Confusion matrix
        a.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in trainSet:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = a(inputs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        matrix_c = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_c, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Tahmin')
        plt.ylabel('Gercek')
        plt.title(f'Confusion Matrix : (Learning Rate:{LR} - {op_name} )')
        plt.savefig(f'ConMatrix_{op_name}-{LR}.png')
        plt.close()


