import os

for i in range(800):
	print('\n\n\n\n\n\n\n',i)
	os.system("ipython dist_rsa/run_iden.py False "+str(i)+' '+str(i+1)+" False")