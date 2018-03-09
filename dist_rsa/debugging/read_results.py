import pickle
import numpy as np

data = pickle.load(open("dist_rsa/data/discrete_sig_search",'rb'))

for sig1,sig2,l1_sig1 in [[0.1,1.0,1.0],[1.0,1.0,1.0],[10.0,1.0,1.0],[100.0,1.0,1.0],
        [1.0,0.1,1.0],[1.0,1.0,1.0],[1.0,10.0,1.0],[1.0,100.0,1.0],
        [1.0,1.0,0.1],[1.0,1.0,1.0],[1.0,1.0,10.0],[1.0,1.0,100.0]]:

    results = data[(sig1,sig2,l1_sig1,"disc",0)]
    print("DISCRETE","sig1:",sig1,"sig2",sig2,"l1_sig1",l1_sig1,results)

    for inf_type in ["var","hmc"]:
    	for i in range(5):

	    	results = [(x,np.exp(y)) for (x,y) in data[(sig1,sig2,l1_sig1,inf_type,i)]]
	    	print(inf_type,"sig1:",sig1,"sig2",sig2,"l1_sig1",l1_sig1,results)