
import csv
import sys
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def get_dataset_tt(dataset):

    trainD=[]
    trainL=[]
    testD=[]
    testL=[]

    csvfile= open('{}/train.csv'.format(dataset)) 
    reader = csv.reader(csvfile)
    attributes = reader.__next__()
    for row in reader:
        row=list(map(float,row))
        trainD.append(row[0:-1])
        trainL.append(int(row[-1]))
    csvfile.close()
    csvfile= open('{}/test.csv'.format(dataset)) 
    reader = csv.reader(csvfile)
    attributes = reader.__next__()
    for row in reader:
        row=list(map(float,row))
        testD.append(row[0:-1])
        testL.append(int(row[-1]))
    csvfile.close()
    csvfile= open('{}/validation.csv'.format(dataset)) 
    reader = csv.reader(csvfile)
    attributes = reader.__next__()
    for row in reader:
        row=list(map(float,row))
        trainD.append(row[0:-1])
        trainL.append(int(row[-1]))
    csvfile.close()
    
    return np.array(trainD), np.array(trainL), np.array(testD), np.array(testL)
def get_dataset(dataset):

    trainD=[]
    trainL=[]

    csvfile= open('{}/train.csv'.format(dataset)) 
    reader = csv.reader(csvfile)
    attributes = reader.__next__()

    for row in reader:
        row=list(map(float,row))
        trainD.append(row[0:-1])
        trainL.append(int(row[-1]))
    csvfile.close()
    
    csvfile= open('{}/test.csv'.format(dataset)) 
    reader = csv.reader(csvfile)
    attributes = reader.__next__()

    for row in reader:
        row=list(map(float,row))
        trainD.append(row[0:-1])
        trainL.append(int(row[-1]))
    csvfile.close()
    
    csvfile= open('{}/validation.csv'.format(dataset)) 
    reader = csv.reader(csvfile)
    attributes = reader.__next__()

    for row in reader:
        row=list(map(float,row))
        trainD.append(row[0:-1])
        trainL.append(int(row[-1]))
    csvfile.close()
    trainD=np.array(trainD)
    trainL=np.array(trainL)
    return trainD,trainL

NUM_FEATURE1=7
NUM_FEATURE2=55


def test_kmeans(data,label,tdata,tlabel, num_classes,n_init=100):
    train_elapsed=time.time()
    kmeans=KMeans(n_clusters=num_classes,n_init=n_init).fit(data)
    train_elapsed=time.time()-train_elapsed
    #print ((kmeans.labels_))
    lout=kmeans.labels_
    llabel=np.ones(num_classes)
    for i in range(num_classes):
        a=np.where(lout==i)[0]
        b=np.where(label==0)[0]
        intersect=np.intersect1d(a,b)
        if (intersect.shape[0]/a.shape[0])>0.5:
            llabel[i]=0
    lout_fin=lout
    for i in range(num_classes):
        lout_fin[lout==i]=llabel[i]
    train_error=sum(lout_fin==label)/label.shape[0]

    test_elapsed=time.time()
    lout=kmeans.predict(tdata)
    test_elapsed=time.time()-test_elapsed
    llabel=np.ones(num_classes)
    for i in range(num_classes):
        a=np.where(lout==i)[0]
        b=np.where(tlabel==0)[0]
        intersect=np.intersect1d(a,b)
        if a.shape[0]>0:
            if (intersect.shape[0]/a.shape[0])>0.5:
                llabel[i]=0
    lout_fin=lout
    for i in range(num_classes):
        lout_fin[lout==i]=llabel[i]
    test_error=sum(lout_fin==tlabel)/tlabel.shape[0]
    #i,train time, train error, test time, test error
    print ("%d,%.4f,%.3f,%.4f,%.3f" % (i,train_elapsed,train_error,test_elapsed,test_error ))


def test_gmm(data,label,tdata,tlabel, num_classes):
    train_elapsed=time.time()
    gmm=GMM(n_components=num_classes).fit(data)
    train_elapsed=time.time()-train_elapsed
    #print ((kmeans.labels_))
    lout=gmm.predict(data)
    llabel=np.ones(num_classes)
    for i in range(num_classes):
        a=np.where(lout==i)[0]
        b=np.where(label==0)[0]
        intersect=np.intersect1d(a,b)
        if (intersect.shape[0]/a.shape[0])>0.5:
            llabel[i]=0
    lout_fin=lout
    for i in range(num_classes):
        lout_fin[lout==i]=llabel[i]
    train_error=sum(lout_fin==label)/label.shape[0]

    test_elapsed=time.time()
    lout=gmm.predict(tdata)
    test_elapsed=time.time()-test_elapsed
    llabel=np.ones(num_classes)
    for i in range(num_classes):
        a=np.where(lout==i)[0]
        b=np.where(tlabel==0)[0]
        intersect=np.intersect1d(a,b)
        if a.shape[0]>0:
            if (intersect.shape[0]/a.shape[0])>0.5:
                llabel[i]=0
    lout_fin=lout
    for i in range(num_classes):
        lout_fin[lout==i]=llabel[i]
    test_error=sum(lout_fin==tlabel)/tlabel.shape[0]
    #i,train time, train error, test time, test error
    print ("%d,%.4f,%.3f,%.4f,%.3f" % (i,train_elapsed,train_error,test_elapsed,test_error ))

def get_pca(data,tdata,num_classes):
    pca=PCA(n_components=num_classes)
    pca=pca.fit(data)
    pdata=pca.fit_transform(data)
    ptdata=pca.fit_transform(tdata)
    return pdata, ptdata

def get_ica(data,tdata,num_classes):
    ica=FastICA(n_components=num_classes)
    ica=ica.fit(data)
    pdata=ica.fit_transform(data)
    ptdata=ica.fit_transform(tdata)
    return pdata,ptdata    

def get_rp(data,tdata,num_classes):
    rp=GaussianRandomProjection(n_components=num_classes)
    pdata=rp.fit_transform(data)
    ptdata=rp.fit_transform(tdata)
    return pdata,ptdata

def get_svd(data,tdata,num_classes):
    svd=TruncatedSVD(n_components=num_classes,n_iter=10)
    svd.fit(data)
    pdata=svd.fit_transform(data)
    ptdata=svd.fit_transform(tdata)
    return pdata,ptdata

if __name__ == "__main__":
    
    data1,label1,tdata1,tlabel1=get_dataset_tt('data_titanic')
    data2,label2,tdata2,tlabel2=get_dataset_tt('data_spam')
    # for i in range(30):
    #     test_kmeans(data1,label1,tdata1,tlabel1,i+1)
    
    # for i in range(30):
    #     test_kmeans(data2,label2,tdata2,tlabel2,i+1)
    
    # for i in range(30):
    #     test_gmm(data1,label1,tdata1,tlabel1,i+1)

    # for i in range(30):
    #     test_gmm(data2,label2,tdata2,tlabel2,i+1)


    # for i in range(30):
    #     for j in range(NUM_FEATURE1):
    #         pdata,ptdata = get_pca(data1,tdata1,j+1)
    #         test_kmeans(pdata,label1,ptdata,tlabel1,i+1)

    # for i in range(30):
    #     for j in range(NUM_FEATURE2):
    #         pdata,ptdata = get_pca(data2,tdata2,j+1)
    #         test_kmeans(pdata,label2,ptdata,tlabel2,i+1)

    # for i in range(10):
    #     for j in range(NUM_FEATURE1):
    #         pdata,ptdata = get_ica(data1,tdata1,j+1)
    #         test_kmeans(pdata,label1,ptdata,tlabel1,i+1)

    # for i in range(10):
    #     for j in range(NUM_FEATURE2):
    #         pdata,ptdata = get_ica(data2,tdata2,j+1)
    #         test_kmeans(pdata,label2,ptdata,tlabel2,i+10)

    # for i in range(10):
    #     for j in range(NUM_FEATURE1):
    #         pdata,ptdata = get_rp(data1,tdata1,j+1)
    #         test_kmeans(pdata,label1,ptdata,tlabel1,i+1)

    # for i in range(10):
    #     for j in range(NUM_FEATURE2):
    #         pdata,ptdata = get_rp(data2,tdata2,j+1)
    #         test_kmeans(pdata,label2,ptdata,tlabel2,i+10)

    for i in range(10):
        for j in range(NUM_FEATURE1-1):
            pdata,ptdata = get_svd(data1,tdata1,j+1)
            test_kmeans(pdata,label1,ptdata,tlabel1,i+1)

    for i in range(10):
        for j in range(NUM_FEATURE2-1):
            pdata,ptdata = get_svd(data2,tdata2,j+1)
            test_kmeans(pdata,label2,ptdata,tlabel2,i+10)
