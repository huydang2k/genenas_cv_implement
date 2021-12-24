import pickle

with open('cifar10.gene_nas.2021-12-23.pkl','rb') as f:
    d = pickle.load(f)
    print(d)