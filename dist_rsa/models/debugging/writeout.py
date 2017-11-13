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
from dist_rsa.models.debugging.l1_cat_short import l1_model as l1_cat_short


if __name__ == "__main__":

    sig1 = 0.1
    sig2 = 0.1
    l1_sig1 = 100.0
    start = 10
    stop = 3000
    qud_num = 10

    out = open("dist_rsa/models/debugging/writeout","w")
    # (best_model,"variational"),
    # (hmc_model,"hmc"),
    for l1_model,model_name in [(l1_cat_short,"short")]:
    # [(best_model,"variational")]:
        out.write("\n\nMODEL:"+model_name+"\n\n")

        for subj,pred in [("man","horse")]:
        # [("woman","horse"),("man","horse"),("horse","man"),("cat","fool")]:
        # metaphors:
            out.write('\n'+subj+","+pred+'\n')
            out.write("L1: utts"+str(0)+str(start))
            worldms=[]
            for x in range(5):
                results,worldm = l1_model((subj,pred,sig1,sig2,l1_sig1,0,start,False,qud_num))
                worldms.append(worldm)
                out.write('\n')
                out.write(str(results[:5]))
                # out.write(str([(x,np.exp(y)) for (x,y) in results[:5]]))
            out.write('\nWORLDS')
            for x in range(5):
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
      


            
            out.write('\nBASELINE:0-100\n')
            results,worldm = l1_model((subj,pred,sig1,sig2,l1_sig1,0,start,True,qud_num))
            out.write(str(results[:10]))

            # out.write('\nBASELINE:100-200\n')
            # results,worldm = l1_model((subj,pred,sig1,sig2,l1_sig1,start,stop,True))
            # out.write(str(results[:5]))
