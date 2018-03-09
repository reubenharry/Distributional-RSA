from __future__ import division
from collections import defaultdict
import scipy
import numpy as np
import pickle
import itertools
from dist_rsa.dbm import *
from dist_rsa.utils.load_data import *
from dist_rsa.utils.helperfunctions import *
from dist_rsa.lm_1b_eval import predict
from dist_rsa.utils.config import abstract_threshold,concrete_threshold
from dist_rsa.utils.distance_under_projection import distance_under_projection
import random
# from dist_rsa.models.debugging.best_model import l1_model as best_model
# from dist_rsa.models.debugging.hmc_model import l1_model as hmc_model
# from dist_rsa.models.debugging.l1_cat_long import l1_model as l1_cat_long
from dist_rsa.models.debugging.l1_cat_short import l1_model as l1_cat_short

name = "0.1"

if __name__ == "__main__":

    sig1 = 0.1
    sig2 = 0.1
    l1_sig1 = 0.1

    start = 10
    stop = 3000
    qud_num = 10

    num_iters = 20

    out = open("dist_rsa/models/debugging/writeout","w")
    # (best_model,"variational"),
    # (hmc_model,"hmc"),
    for l1_model,model_name in [(l1_cat_short,"short")]:
    # [(best_model,"variational")]:
        out.write("\n\nMODEL:"+model_name+"\n\n")

        for subj,pred in [("subj1","pred1"),("subj2","pred1")]:
        # [("woman","horse"),("man","horse"),("horse","man"),("cat","fool")]:
        # metaphors:
            out.write('\n'+subj+","+pred+'\n'+"l0_sig1:"+str(sig1)+"l0_sig2:"+str(sig2)+"l1_sig1"+str(l1_sig1)+"\n")
            out.write("L1: utts"+str(0)+str(start))
            worldms=[]
            for x in range(num_iters):
                results,worldm = l1_model((subj,pred,sig1,sig2,l1_sig1,0,start,False,qud_num,x))
                worldms.append(worldm)
                out.write('\n')
                out.write("Iter "+str(x)+" "+str(results[:5]))
                # out.write(str([(x,np.exp(y)) for (x,y) in results[:5]]))
            out.write('\nWorld movement along each qud')
            for x in range(num_iters):
                out.write('\n')
                out.write(str(worldms[x][:5]))
            out.write('\n')

            # out.write("L1: utts"+str(start)+str(stop))
            # worldms=[]
            # for x in range(3):
            #     results,worldm = l1_model((subj,pred,sig1,sig2,l1_sig1,start,stop,False))
            #     worldms.append(worldm)
            #     out.write('\n')
            #     out.write(str(results[:5]))
            # out.write('\nWORLDS')
            # for x in range(3):
            #     out.write('\n')
            #     out.write(str(worldms[x][:5]))
            # out.write('\n')
      


            
            # out.write('\nBASELINE:0-100\n')
            # results,worldm = l1_model((subj,pred,sig1,sig2,l1_sig1,0,start,True,qud_num,1000))
            # out.write(str(results[:10]))

            # out.write('\nBASELINE:100-200\n')
            # results,worldm = l1_model((subj,pred,sig1,sig2,l1_sig1,start,stop,True))
            # out.write(str(results[:5]))
