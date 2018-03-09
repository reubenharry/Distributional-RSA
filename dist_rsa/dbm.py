from __future__ import division
import scipy
import numpy as np
import pickle
import itertools
import nltk
from dist_rsa.rsa.tensorflow_l1 import tf_l1
from dist_rsa.rsa.tensorflow_l1_mixture import tf_l1 as tf_l1_mixture
from dist_rsa.rsa.tensorflow_s2 import tf_s2
from dist_rsa.rsa.tensorflow_l1_with_trivial import tf_l1_with_trivial
from dist_rsa.rsa.tensorflow_l1_only_trivial import tf_l1_only_trivial
from dist_rsa.rsa.tensorflow_l1_baseline import tf_l1_baseline
from dist_rsa.rsa.tensorflow_l1_noncat import tf_l1_noncat
from dist_rsa.rsa.tensorflow_l1_discrete import tf_l1_discrete
from dist_rsa.rsa.tensorflow_l1_discrete_only_trivial import tf_l1_discrete_only_trivial
from dist_rsa.utils.refine_vectors import h_dict,processVecMatrix
# from dist_rsa.utils.load_data import 
from dist_rsa.utils.helperfunctions import *



class Inference_Params:
    def __init__(
        self,
        vecs,
        subject,predicate,
        quds,possible_utterances,
        sig1,sig2,
        qud_weight,freq_weight,
        categorical,
        sample_number,
        number_of_qud_dimensions,
        seed,
        trivial_qud_prior,
        l1_sig1,
        burn_in=0,
        step_size=0.25,
        poss_utt_frequencies=None,
        qud_frequencies=None,
        qud_prior_weight=0.5,
        rationality=1.0,
        norm_vectors=False,
        variational=True,
        variational_steps=100,
        baseline=False,
        only_trivial=False,
        discrete_l1=False,
        resolution=(100,0.1),
        just_s1=False,
        just_l0=False,
        target_qud=None,
        mixture_variational=False,
        ):


            self.l1_sig1=l1_sig1
            self.vecs=vecs
            self.subject = subject
            self.subject_vector = np.mean([self.vecs[x] for x in self.subject],axis=0)
            self.predicate = predicate
            self.quds=quds
            self.possible_utterances=possible_utterances
            self.sig1=sig1
            self.sig2=sig2
            self.model_type=categorical
            self.sample_number=sample_number
            self.burn_in=burn_in
            self.number_of_qud_dimensions=number_of_qud_dimensions
            self.trivial_qud_prior=trivial_qud_prior
            self.qud_prior_weight=qud_prior_weight
            self.rationality=rationality
            self.poss_utt_frequencies=poss_utt_frequencies
            self.qud_frequencies = qud_frequencies
            self.qud_weight=qud_weight
            self.freq_weight=freq_weight
            self.seed=seed
            self.step_size=step_size
            self.vec_length = self.vecs["the"].shape[0]
            self.variational=variational
            self.mixture_variational=mixture_variational
            self.variational_steps=variational_steps
            self.only_trivial=only_trivial
            self.baseline=baseline
            self.discrete_l1=discrete_l1
            self.resolution=resolution
            self.just_s1=just_s1
            self.just_l0=just_l0
            self.target_qud=target_qud
            
            if norm_vectors:
                for vec in vecs:
                    vecs[vec] = vecs[vec] / np.linalg.norm(vecs[vec])

            for word in self.quds+self.possible_utterances:
                if word not in vecs:
                    if word in self.quds:
                        self.quds.remove(word)
                    elif word in self.possible_utterances:
                        self.possible_utterances.remove(word)



class Dist_RSA_Inference:
    def __init__(self,inference_params):
        self.inference_params = inference_params

    def compute_l1(self,load,save=True):

        print("subject:",self.inference_params.subject)
        print("predicate",self.inference_params.predicate)
        print("SIGs 1&2:",self.inference_params.sig1,self.inference_params.sig2)
        print("L1 SIG",self.inference_params.l1_sig1)
        print("step_size",self.inference_params.step_size)
        print("utt weight, qud weight",self.inference_params.freq_weight,self.inference_params.qud_weight)
        print("number of qud dimensions:",self.inference_params.number_of_qud_dimensions)
        print("trival qud prior on?",self.inference_params.trivial_qud_prior)
        print("rationality:",self.inference_params.rationality)
        print("sample number",self.inference_params.sample_number)

        
        message = "Running "+self.inference_params.model_type+" RSA with "+str(len(self.inference_params.possible_utterances))+" possible utterances and " + str(len(self.inference_params.quds))
        print(message)


        if self.inference_params.discrete_l1:
            print("RUNNING DISCRETE MODEL")
            if self.inference_params.only_trivial:
                tf_results = tf_l1_discrete_only_trivial(self.inference_params)
            else: 
                tf_results = tf_l1_discrete(self.inference_params)
            self.tf_results=tf_results
            return None

        elif self.inference_params.only_trivial:
            print("RUNNING MODEL WITHOUT QUDS")
            tf_results = tf_l1_only_trivial(self.inference_params)
            self.world_samples = tf_results
            return None

        elif self.inference_params.model_type=='categorical':

            print("is baseline?",self.inference_params.baseline)
            if self.inference_params.baseline:  
                print("RUNNING BASELINE MODEL")
                # tf_results = tf_l1_baseline(self.inference_params)  
                self.qud_samples = tf_l1_baseline(self.inference_params)  
                return None

            elif self.inference_params.trivial_qud_prior:
                print("RUNNING CAT WITH TRIVIAL MODEL")
                tf_results = tf_l1_with_trivial(self.inference_params)
            elif self.inference_params.mixture_variational:
                print("RUNNING MIXTURE VARIATIONAL MODEL")
                tf_results = tf_l1_mixture(self.inference_params)
            else:
                print("RUNNING CAT WITHOUT TRIVIAL MODEL")
                tf_results = tf_l1(self.inference_params)
    


        elif self.inference_params.model_type=='non-categorical':
            print("RUNNING NONCAT WITHOUT TRIVIAL MODEL")
            tf_results = tf_l1_noncat(self.inference_params)
            
        self.qud_samples,self.world_samples = tf_results

    def compute_s1(self,s1_world,world_movement=False,debug=False,vectorization=1):
        from dist_rsa.rsa.tensorflow_s1 import tf_s1
        from dist_rsa.rsa.tensorflow_s1_old import tf_s1_old
        from dist_rsa.rsa.tensorflow_s1_triple_vec import tf_s1_triple_vec
        # from dist_rsa.rsa.tensorflow_s1_old import tf_s1

        print("possible_utterances",self.inference_params.possible_utterances)

        s1_world = tf.cast(s1_world,dtype=tf.float32)

        poss_utt_freqs = np.log(np.array([lookup(self.inference_params.poss_utt_frequencies,x,'NOUN') for x in self.inference_params.possible_utterances]))
        self.inference_params.weighted_utt_frequency_array=weighted_utt_frequency_array = poss_utt_freqs-scipy.misc.logsumexp(poss_utt_freqs)
        self.inference_params.poss_utts=tf.cast(as_a_matrix(self.inference_params.possible_utterances,self.inference_params.vecs),dtype=tf.float32)
        qud_combinations = combine_quds(self.inference_params.quds,self.inference_params.number_of_qud_dimensions)
        self.inference_params.qud_matrix=tf.cast((np.asarray([np.asarray([self.inference_params.vecs[word] for word in words]).T for words in qud_combinations])),dtype=tf.float32)
        self.inference_params.listener_world=tf.cast(self.inference_params.subject_vector,dtype=tf.float32) 
        print("computing s1")
        # tv = tf_s1_triple_vec(self.inference_params,s1_world=s1_world,world_movement=world_movement,debug=debug)
        # import edward as ed
        # print("triple vec results",ed.get_session().run(tv))
        
        #NB ACTUALLY VECTORIZATION 1 and 2 are both 2, i think
        if vectorization==1: self.s1_results=tf_s1_old(self.inference_params,s1_world=s1_world,world_movement=world_movement,debug=debug)
        elif vectorization==2: self.s1_results=tf_s1(self.inference_params,s1_world=s1_world,world_movement=world_movement,debug=debug)
        elif vectorization==3: self.s1_results=tf_s1_triple_vec(self.inference_params,s1_world=s1_world,world_movement=world_movement,debug=debug)
    def compute_s2(self,s2_world,s2_qud):

        s2_world = tf.cast(s2_world,dtype=tf.float32)
        if self.baseline:
            
            self.s2_results = tf_s2_qud_only(self.inference_params,s2_world,s2_qud)

        self.s2_results=tf_s2(self.inference_params,s2_world,s2_qud)



    def world_movement(self,metric,comparanda,do_projection=False):

        vecs = self.inference_params.vecs

        if metric != 'cosine':
            raise Exception("Only implemented for metric=cosine")

        if do_projection and self.inference_params.model_type=="categorical":
            # raise Exception("Can't project onto QUD in categorical L1 model, because, like, which QUD")

            dists=[]
            for word in comparanda:
                # print((self.subject_vector).shape)
                subject = projection_debug(self.inference_params.subject_vector,np.expand_dims(vecs[word],-1))
                observation = np.expand_dims(projection_debug(np.mean(self.world_samples,axis=0),np.expand_dims(vecs[word],-1)),0)
                dists.append((word,observation-subject))
                # dists.append((word,observation-subject))
            return sorted(dists,key=lambda x:x[1],reverse=True)



        # if do_projection:
        #     subject = projection(self.subject_vector,np.mean(self.qud_samples,axis=0))
        #     observation = np.expand_dims(projection(np.mean(self.world_samples,axis=0),np.mean(self.qud_samples,axis=0)),0)
        subject = self.inference_params.subject_vector
        observation = self.world_samples

        words_with_distance_to_prior = np.asarray(list(map(lambda x: scipy.spatial.distance.cosine(vecs[x],subject),comparanda)))

        words_with_distance_to_posterior = np.asarray(list(map((lambda y:   list(map((lambda x: scipy.spatial.distance.cosine(vecs[x],y)),comparanda))),observation)))
        
        word_movement = words_with_distance_to_posterior-words_with_distance_to_prior

        mean_word_movement = np.squeeze(np.mean(word_movement,axis=0))

        out = sorted(list(zip(comparanda,mean_word_movement)),key=lambda x:abs(x[1]),reverse=True)

        return out

       
    def qud_results(self,comparanda=None):

        # new_vecs = defaultdict(lambda x : np.zeros(self.inference_params.vec_length))
        # for x in self.inference_params.vecs:
        #     new_vecs[x] = self.inference_params.vecs[x]

        if self.inference_params.model_type=="categorical":
            if self.inference_params.trivial_qud_prior:
                for i,result in enumerate(self.qud_samples):
                    if result[0] != ['TRIVIAL']:
                        if i < 10:
                            print(result[0],np.exp(result[1]))
                    else:
                        print(result[0],np.exp(result[1]))
                        print(i)
                        break
                print("LINE 259 of dbm",self.qud_samples[:10])
                print(list(zip(*self.qud_samples))[0])
                return list(zip(*self.qud_samples))[0].index(["TRIVIAL"])

            else:
                return self.qud_samples
        # elif self.inference_params.model_type=="non-categorical":
        #     if comparanda==None:
        #         comparanda=self.inference_params.quds
        #     nearest = nearest_neighbours(support=comparanda,comp_point=np.expand_dims(np.mean(self.qud_samples,axis=0),0),vecs=new_vecs)
        #     return list(zip(*nearest))[:20]
