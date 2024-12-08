import torchvision as tv
import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 10
batch_size = 128
#Classe contenant le modèle 
class my_class(nn.Module):
    def __init__(self,hidden):
        super().__init__()

        #Première couche à 3 * 32 * 32 afin de prendre les 3 canaux de tous les pixels d'une image à la fois, pas de perte
        self.a = nn.Linear(3*32*32,hidden)
        self.b = nn.Linear(hidden,hidden)
        self.c = nn.Linear(hidden,hidden)
        self.d = nn.Linear(hidden,nclasses)

    def forward(self,x):
        return self.d(self.c((self.b((self.a(x))))))

#Fonction de calcul de précision de prédiction
def accuracy(labels,predicted_labels):
    accu = 0
    for i in range(len(labels)):
        if labels[i] == predicted_labels[i]:
            #print(str(labels[i])+" equals to "+str(predicted_labels[i]))
            accu += 1

    return accu/batch_size * 100

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
        
            
            batch = batch.permute(0,2,3,1)
            
            batch = batch.reshape(-1,32*32*3)
            
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

            batch = batch.permute(0,2,3,1)
            
            batch = batch.reshape(-1,32*32*3)
            # Validation sur des batchs de 32 images
            out= myclass(batch)

            loss = lossFn(out, label)
            index_max = torch.max(out,1)[1]

            acc_cum = acc_cum + accuracy(index_max,label)

        print(f"Accuracy at epoch n°{_} is {round(acc_cum/len(iter_valid),3)} loss is {loss}")

def main():
#Init du modèle 
    myclass = my_class(32)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    myclass.to(dev)
    #Init des fonctions 
    tensor = tv.transforms.ToTensor()
    


    #Chargement des données 
    train_set = tv.datasets.CIFAR10("." ,download=True ,train=True, transform=tensor)
    valid_set = tv.datasets.CIFAR10(".", train=False,download=True, transform=tensor)

    #transformation en tensor, format [Nb_batches,Channels,largeur,hauteur]
    data_train_load = torch.utils.data.DataLoader(train_set,batch_size=128,shuffle=True)
    data_valid_load = torch.utils.data.DataLoader(valid_set,batch_size=128,shuffle=True )


    

    fit_one_cycle(10,data_train_load,data_valid_load,myclass)

if __name__ == "__main__":
    main()
  
