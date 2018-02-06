# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:59:00 2018

@author: Yossi
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

from scipy.spatial.distance import pdist, squareform

import scipy.sparse as sparse
import pymc as pm
import AffinityClustering as AC


from sklearn.cluster import KMeans
LIB='./data/'
def DistanceMatrix(X):
    X[np.isnan(X)]=0
    D = pdist(X, metric='sqeuclidean')
    D = squareform(D)
    return D


def AffinityMatrix(X,k=10):
    #The distance matrix
    D = DistanceMatrix(X)
    """
    #The Affinity matrix
    A=np.zeros(D.shape)
    sigma=X.std(axis=1)
    sigma[sigma==0]=sigma[sigma>0].min()
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            A[i,j]=np.exp(-D[i,j]/(sigma[i]*sigma[j]))
    """
    #constract the Laplacian matrix
    A=D
    Q=np.zeros(A.shape)
    for i in range(Q.shape[0]):
        Q[i,i]=np.sqrt(A[i,:].sum())
    
    Q=np.matrix(Q)
    Q=Q.I
    L=Q*A*Q
    
    ######K-means
    vals, vecs = sparse.linalg.eigs(L, k, which='LM')
    vals=vals.real
    vecs=vecs.real
    
    return vals,vecs
##################################################
#Clustering the airport using the distance matrix
def clustering(cancelled,k=7):
    from sklearn.mixture import GaussianMixture

    # The eigen vectors of the distance matrix
    vals,vecs=AffinityMatrix(cancelled.values, k)
    
    gmm = GaussianMixture(n_components=k).fit(vecs)
    labels = gmm.predict(vecs)    
    probs = gmm.predict_proba(vecs)
    
    return labels, probs

##################################################
# Finding the mean of the disrupted flight using MCMC

def lambdaMCMC(dg):
    alpha=1.0/(dg.mean() + (dg.mean()==0))  #alpha is an hyperparameter that gives an intial guess for lambda=1/alpha
    lambda1=pm.Exponential("lambda1", alpha) #prior deistribution
    #Likelihood
    observation=pm.Poisson("obs", lambda1, value=dg, observed=True)
    model=pm.Model([observation,lambda1])
    mcmc=pm.MCMC(model)
    mcmc.sample(40000,10000)
    lambda1samples=mcmc.trace('lambda1')[:]
    return lambda1samples.mean(), lambda1samples.std()
##################################################
if __name__ == "__main__":
    cancelled=pd.read_csv(LIB+'cancelled.csv', index_col='FL_DATE')
    cancelled=cancelled.transpose()
    
    counts=pd.read_csv(LIB + 'counts.csv',index_col='FL_DATE')
    counts=counts.transpose()

    #total flight cancelled
    cnc=pd.DataFrame(counts.values*cancelled.values, columns=counts.columns, index=counts.index)
    
    #How many component? using cross-validation to choose the amount of components
    if False:
        k=50
        vals,vecs=AffinityMatrix(cancelled,k)
    
        """
        kmeans = KMeans(n_clusters=k).fit(vecs)
    
        kmeans.labels_    #label each airport to cluster
        """
        #####################################
        
        #####Gaussian Mixture model  ########
        
        gmm = GaussianMixture(n_components=k).fit(vecs)
        labels = gmm.predict(vecs)
        
        probs = gmm.predict_proba(vecs)
    
        n_components = np.arange(1, 50)
        models = [GaussianMixture(n, covariance_type='diag', random_state=0).fit(vecs)
            for n in n_components]
        #Baysian information criterion
        plt.plot(n_components, [m.bic(vecs) for m in models], label='BIC')
        plt.plot(n_components, [m.aic(vecs) for m in models], label='AIC')
        plt.legend(loc='best')
        plt.xlabel('n_components');
    
    
    labels,probs = clustering(cancelled,k=7)
    airportsloc=pd.read_csv('./data/BackupData/airportsLoc.csv',index_col=0)
    temp=pd.DataFrame(np.array([labels,np.max(probs,axis=1)]).T, index=cancelled.index, columns=['group','prob'])
    AllAirports=pd.concat([airportsloc, temp], axis=1)
    
    AllAirports['count']=counts.fillna(0).mean(axis=1)
    AllAirports['name']=airportsloc['name']

    
    #Calculate the mean of disrupted fligth using MCMC
    for i,row in cnc.iterrows():
        m,std=lambdaMCMC(row.dropna())        
        AllAirports.loc[i,'lambdaM']=m
        AllAirports.loc[i,'lambdaSTD']=std
    
    
    
    AllAirports.to_csv(LIB+'allairports.csv',encoding='utf-8')
    
    
    


