import numpy as np
import time
from matplotlib import pyplot as plt
import math
from sklearn.model_selection import ParameterGrid
from numpy import matlib
from ucimlrepo import fetch_ucirepo 


def split_train_test(X, y, train_ratio = 0.9, seed = 42):
    np.random.seed(seed)

    n_instances = len(X)
    ntr = round(n_instances*train_ratio)

    indices = np.arange(n_instances)
    np.random.shuffle(indices)

    Xtr = X[indices[:ntr]]
    ytr = y[indices[:ntr]]

    Xtst = X[indices[ntr:]]
    ytst = y[indices[ntr:]]

    return Xtr, ytr, Xtst, ytst


def load_dataset(id):  
    # fetch dataset 
    dataset = fetch_ucirepo(id=id) 
    
    # data (as pandas dataframes) 
    X = dataset.data.features 
    y =dataset.data.targets 
    
    # dictionary gthering infos about the metadata (url, abstract, ... etc.)
    metadata_infos_dict = dataset.metadata
    print('data url:\n', metadata_infos_dict['data_url'])
    
    # variable information
    var_infos = dataset.variables.to_numpy()
    
    data_vectors = X.to_numpy() #instance vectors with features
    features_names = X.columns.to_numpy() #getting the names of each feature
    
    data_labels = y.to_numpy() #output labels for each instance
    label_name = y.columns.to_numpy() # name of the output label
    
    return data_vectors, features_names, data_labels, label_name



def calculate_accuracy(confusion_matrix):

    correct_predictions = np.trace(confusion_matrix)
    #print(correct_predictions)
    total_predictions = np.sum(confusion_matrix)
    #print(total_predictions)
    # Calculating accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy


def getEuclideanDistance(single_point,array):
    nrows, ncols, nfeatures=array.shape[0],array.shape[1], array.shape[2]
    points=array.reshape((nrows*ncols,nfeatures))
                         
    dist = (points - single_point)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    dist=dist.reshape((nrows,ncols))
    return dist
def optimize_params(params_grid,X,Y,X_test=[],Y_test = []) : 
    best_score = 0
    cmp = 0
    for params in ParameterGrid(params_grid):
        cmp+=1
        print("\rOptimization : "+str(int(cmp*100/len(ParameterGrid(params_grid))))+"%",end="",flush=True)
        som = SOM(len(X[0]),params['n_dims'], params['n_epochs'], params['eta0'], params['eta_decay'], params['sgm0'], params['sgm_decay'])
        som.train_SOM(X)
        Confusion_Matrix = som.confusion_matrix(X_test,Y_test)
        accuracy = calculate_accuracy(Confusion_Matrix)

    # Calculer un score à partir de la matrice de confusion
        if accuracy > best_score:
            best_score = accuracy
            best_params = params
    return (best_params,best_score)
    
class SOM() : 
    def __init__(self,nfeatures,ndims=10,nepochs=10,eta0=0.1,etadecay=0.05, sgm0=20, sgmdecay=0.05) : 
        self.nepochs=nepochs
        self.nrows = ndims
        self.ncols = ndims
        self.eta0=eta0
        self.etadecay=etadecay
        self.sgm0=sgm0
        self.sgmdecay=sgmdecay
        self.mu = 0
        self.sigma=0.1
        self.nfeatures = nfeatures
        self.som = np.random.normal(self.mu, self.sigma, (self.nrows,self.ncols,self.nfeatures))

    def train_SOM(self,X) : 
        #Generate coordinate system
        x,y=np.meshgrid(range(self.ncols),range(self.nrows))
        nfeatures=X.shape[1]
        ntrainingvectors=X.shape[0]
        
        for t in range (1,self.nepochs+1):
            #print("\rEpoch : "+str(t),end="",flush=True)
            #Compute the learning rate for the current epoch
            eta = self.eta0 * math.exp(-t*self.etadecay)
            
            #Compute the variance of the Gaussian (Neighbourhood) function for the ucrrent epoch
            sgm = self.sgm0 * math.exp(-t*self.sgmdecay)
            
            #Consider the width of the Gaussian function as 3 sigma
            width = math.ceil(sgm*3)
            
            for ntraining in range(ntrainingvectors):
                trainingVector = X[ntraining,:]
                
                # Compute the Euclidean distance between the training vector and
                # each neuron in the SOM map
                dist = getEuclideanDistance(trainingVector, self.som)
        
                # Find 2D coordinates of the Best Matching Unit (bmu)
                bmurow, bmucol =np.unravel_index(np.argmin(dist, axis=None), dist.shape) 
                
                
                #Generate a Gaussian function centered on the location of the bmu
                g = np.exp(-((np.power(x - bmucol,2)) + (np.power(y - bmurow,2))) / (2*sgm*sgm))
                #Determine the boundary of the local neighbourhood
                fromrow = max(0,bmurow - width)
                torow   = min(bmurow + width,self.nrows)
                fromcol = max(0,bmucol - width)
                tocol   = min(bmucol + width,self.ncols)

                
                #Get the neighbouring neurons and determine the size of the neighbourhood
                neighbourNeurons = self.som[fromrow:torow,fromcol:tocol,:]
                sz = neighbourNeurons.shape
                
                #Transform the training vector and the Gaussian function into 
                # multi-dimensional to facilitate the computation of the neuron weights update
                T = np.matlib.repmat(trainingVector,sz[0]*sz[1],1).reshape((sz[0],sz[1],nfeatures))                  
                G = np.dstack([g[fromrow:torow,fromcol:tocol]]*nfeatures)

                # Update the weights of the neurons that are in the neighbourhood of the bmu
                neighbourNeurons = neighbourNeurons + eta * G * (T - neighbourNeurons)

                
                #Put the new weights of the BMU neighbouring neurons back to the
                #entire SOM map
                self.som[fromrow:torow,fromcol:tocol,:] = neighbourNeurons
    def plot_SOM(self,resolution) : 
        
        fig, ax=plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=(15,15))
        
        for k in range(self.nrows):
            for l in range (self.ncols):
                A=self.som[k,l,:].reshape((resolution[0],resolution[1]))
                ax[k,l].imshow(A,cmap="plasma")
                ax[k,l].set_yticks([])
            ax[k,l].set_xticks([]) 
        
    def confusion_matrix(self,X,Y) : 
        classes = []
        different_labels = []
        
        for i in Y : 
            if i not in different_labels : 
                different_labels.append(i)
        for i in Y : 
            classes.append(different_labels.index(i)+1)

        grid_=np.zeros((self.nrows,self.ncols))
        nclasses=np.max(classes)
        confusion_matrix=np.zeros((nclasses,nclasses))
        self.different_labels = different_labels
        nclasses=np.max(classes)

        som_cl=np.zeros((self.nrows,self.ncols,nclasses+1))
        nfeatures=X.shape[1]
        ntrainingvectors=X.shape[0]
        
        for ntraining in range(ntrainingvectors):
            trainingVector = X[ntraining,:]
            class_of_sample= classes[ntraining]    
            # Compute the Euclidean distance between the training vector and
            # each neuron in the SOM map
            dist = getEuclideanDistance(trainingVector, self.som)
        
            # Find 2D coordinates of the Best Matching Unit (bmu)
            bmurow, bmucol =np.unravel_index(np.argmin(dist, axis=None), dist.shape) 
            
            
            som_cl[bmurow, bmucol,class_of_sample]=som_cl[bmurow, bmucol,class_of_sample]+1
        
        
        
        for i in range (self.nrows):
            for j in range (self.ncols):
                grid_[i,j]=np.argmax(som_cl[i,j,:])

    
        for ntraining in range(ntrainingvectors):
            trainingVector = X[ntraining,:]
            class_of_sample= classes[ntraining]    
            # Compute the Euclidean distance between the training vector and
            # each neuron in the SOM map
            dist = getEuclideanDistance(trainingVector, self.som)
        
            # Find 2D coordinates of the Best Matching Unit (bmu)
            bmurow, bmucol =np.unravel_index(np.argmin(dist, axis=None), dist.shape) 
            
            predicted=np.argmax(som_cl[bmurow, bmucol,:])
            confusion_matrix[class_of_sample-1, predicted-1]=confusion_matrix[class_of_sample-1, predicted-1]+1
        return confusion_matrix.astype(int)