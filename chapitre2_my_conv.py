import torchvision as tv
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import perf_counter

nclasses = 10
b_size = 128
epoch = 5

class custom_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
    def forward(self,x):
        return self.activation(self.bn(self.conv(x)))
#Classe contenant le modèle 
class my_class(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.a = custom_conv(in_channels,out_channels)
        self.b = custom_conv(out_channels,out_channels)
        self.c = custom_conv(out_channels,out_channels)

        self.residual = nn.Conv2d(in_channels,out_channels,kernel_size=2, stride=1, padding=0)
        self.residual2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.d = nn.Linear(28800,nclasses)

        self.pooling = nn.MaxPool2d(2,stride=1)

    def forward(self,x):
        identity = x 

        a = self.a(x)

        a = self.pooling(a)
        
        b = self.b(a)

        #b = b + self.residual(identity)
        b = self.pooling(b)

        c = self.c(b)
        #black magic

        #c = c + self.residual2(identity)

        c = c.view(c.size(0), -1)

        d = self.d(c)

        return d
#Fonction de calcul de précision de prédiction
def accuracy(labels,predicted_labels):
    accu = 0
    for i in range(len(labels)):
        if labels[i] == predicted_labels[i]:
            #print(str(labels[i])+" equals to "+str(predicted_labels[i]))
            accu += 1

    return accu/b_size * 100

#Fonction d'entrainement, paramètre epoch : entier définissant le nb de fois que l'on itère sur le dataset
def fit_one_cycle(epoch,data_train_load,data_valid_load,myclass):

    #inits
    lossFn = F.cross_entropy
    learningRate = 0.0001


    opt = torch.optim.Adam(myclass.parameters() ,lr=learningRate)

    
    for _ in range(epoch):
        iter_train  = iter(data_train_load)

        #Init du mode entrainement
        myclass.train()

        #Batch est du format : [Num_images,channel_rgb, height, width]
        for batch,label in iter_train:
            #reset gradients to 0
            opt.zero_grad()

            batch = batch.to('cuda')
            label = label.to('cuda')

            out = myclass(batch)                
            #Compute loss
            loss = lossFn(out, label)

            #compute gradients
            loss.backward() 
                
            #update parameters
            opt.step()

                

        #Etape de validation 
        myclass.eval()
        iter_valid = iter(data_valid_load)

        acc_cum = 0
        for batch,label in iter_valid:
            
            batch = batch.to('cuda')
            label = label.to('cuda')
            
            # Validation sur des batchs de 32 images
            out= myclass(batch)



            loss = lossFn(out, label)
            index_max = torch.max(out,1)[1]

            acc_cum = acc_cum + accuracy(index_max,label)

        print(f"Accuracy at epoch n°{_} is {round(acc_cum/len(iter_valid),3)} loss is {loss}")

def main():
#Init du modèle 
    
    myclass = my_class(3,32)

    #Init des fonctions 
    tensor = tv.transforms.ToTensor()
    
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    myclass.to(dev)


    #Chargement des données 
    train_set = tv.datasets.CIFAR10("." ,download=True ,train=True, transform=tensor)
    valid_set = tv.datasets.CIFAR10(".", train=False,download=True, transform=tensor)

    #transformation en tensor, format [Nb_batches,Channels,largeur,hauteur]
    data_train_load = torch.utils.data.DataLoader(train_set,batch_size=b_size,shuffle=True)
    data_valid_load = torch.utils.data.DataLoader(valid_set,batch_size=b_size,shuffle=True )


    
    t = perf_counter()
    fit_one_cycle(epoch,data_train_load,data_valid_load,myclass)
    t2 = perf_counter()

    print(f"Time to train on {epoch} epochs is {t2-t} s")

if __name__ == "__main__":
    main()
  