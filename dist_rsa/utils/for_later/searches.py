got skinny for man is oak on huge set:
run = DistRSAInference(
            subject=subj,predicate=pred,
            # possible_utterances=animals,
            # quds=animal_features,
            quds=list(set(['strong','stable','wooden','connected','leafy','unyielding']+list(list(zip(*visualize_cosine(np.mean([vecs[pred],vecs[subj]],axis=0),[x for x in adjectives if x in vecs],vecs)[:500]))[0]))),
            # quds = animal_features,
        #         ['unyielding']+list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),freq_sorted_adjs,vecs)[:500:10]))[0]),

            # possible_utterances=sorted_nouns[:1000]+[pred],
            possible_utterances=sorted_nouns[:1000]+sorted_adjs[:1000]+[pred],
            # sorted_nouns[sorted_nouns.index(pred) if pred in sorted_nouns else 500]+['horse'],
            object_name="animals_spec",
            mean_vecs=True,
            pca_remove_top_dims=False,
            sig1=0.001,sig2=0.1,
        #         proposed_sig2*scaling_factor,
            qud_weight=0.0,freq_weight=1.0,
            categorical="categorical",
            vec_length=50,vec_type="glove.6B.",
            sample_number = 750,
            number_of_qud_dimensions=1,
            burn_in=500,
            seed=False,trivial_qud_prior=False,
            step_size=0.0005,
            frequencies=prob_dict,
            qud_prior_weight=0.5,
            rationality=0.99,
            run_s2=False,
            speaker_world=vecs[subj]+(1/10*vecs["unyielding"]),
            s1_only=False
            )

good l1: 
	run = DistRSAInference(
            subject=subj,predicate=pred,
            # possible_utterances=animals,
            quds=animal_features,
            # quds = animal_features,
        #         ['unyielding']+list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),freq_sorted_adjs,vecs)[:500:10]))[0]),

            possible_utterances=sorted_nouns[:1000]+[pred],
            # sorted_nouns[sorted_nouns.index(pred) if pred in sorted_nouns else 500]+['horse'],
            object_name="animals_spec",
            mean_vecs=True,
            pca_remove_top_dims=False,
            sig1=10.0,sig2=0.1,
        #         proposed_sig2*scaling_factor,
            qud_weight=0.0,freq_weight=0.0,
            categorical="categorical",
            vec_length=50,vec_type="glove.6B.",
            sample_number = 500,
            number_of_qud_dimensions=1,
            burn_in=400,
            seed=False,trivial_qud_prior=False,
            step_size=0.25,
            frequencies=prob_dict,
            qud_prior_weight=0.5,
            rationality=0.7,
            run_s2=False,
            speaker_world=vecs[subj]+(1/10*vecs["unyielding"]),
            s1_only=True
            )

functional_noncat = noncat = DistRSAInference(
	subject=['man'],predicate=animal,
	quds=animal_features,possible_utterances=[animal]+nouns[50:1000],
	object_name="animals_non_cat",
	mean_vecs=True,
	pca_remove_top_dims=False,
	sig1=10.0,sig2=0.04,
	qud_weight=0.0,freq_weight=1.0,
	categorical="non-categorical",
	vec_length=50,vec_type="glove.6B.",
	sample_number = 100000,
	number_of_qud_dimensions=1,
	burn_in=75000,
	seed=False,trivial_qud_prior=False,
	crop = 100000,
	step_size=0.03
)

    good_cat = DistRSAInference(
    subject=['man'],predicate=word,
    # quds=list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),sorted_adjs)[:500]))[0]),
    possible_utterances=sorted_nouns[:sorted_nouns.index(word)+1],

    quds=['strong','stable','wooden','connected','leafy','unyielding'],
    # possible_utterances=nouns[:1000]+adjectives[:1000],
    object_name="animals_spec",
    mean_vecs=True,
    pca_remove_top_dims=False,
    sig1=10.0,sig2=0.1,
    qud_weight=0.0,freq_weight=1.0,
    categorical="categorical",
    vec_length=50,vec_type="glove.6B.",
    sample_number = 1000,
    number_of_qud_dimensions=1,
    burn_in=900,
    seed=False,trivial_qud_prior=False,
    step_size=0.01,
    frequencies=prob_dict,
    )

           great run = DistRSAInference(
        subject=['man'],predicate=word,
        # quds=list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),sorted_adjs)[:500]))[0]),
        possible_utterances=sorted_nouns[:sorted_nouns.index(word)+1],
        quds=['strong','stable','wooden','connected','leafy','unyielding'],
        # possible_utterances=nouns[:1000]+adjectives[:1000],
        object_name="animals_spec",
        mean_vecs=True,
        pca_remove_top_dims=False,
        sig1=10.0,sig2=1.0,
        qud_weight=0.0,freq_weight=1.0,
        categorical="categorical",
        vec_length=50,vec_type="glove.6B.",
        sample_number = 100,
        number_of_qud_dimensions=1,
        burn_in=90,
        seed=False,trivial_qud_prior=False,
        step_size=0.01,
        frequencies=prob_dict,
        )

               long_run = DistRSAInference(
            subject=['man'],predicate=animal,
            quds=list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs['lion']],axis=0),sorted_adjs)[:500]))[0]),
            possible_utterances=sorted_nouns[:sorted_nouns.index(animal)+1],
            object_name="animals_spec",
            mean_vecs=True,
            pca_remove_top_dims=False,
            sig1=10.0,sig2=0.05,
            qud_weight=0.0,freq_weight=1.0,
            categorical="categorical",
            vec_length=50,vec_type="glove.6B.",
            sample_number = 2000,
            number_of_qud_dimensions=1,
            burn_in=1000,
            seed=False,trivial_qud_prior=False,
            step_size=0.1,
            frequencies=prob_dict
            )

for word in ["tree"]:
    
    subject = ["the", "man", "is", "a"]
    
    distance  = scipy.spatial.distance.cosine(vecs["man"],vecs[word])
    print(distance*scaling_factor)
    #prob_dict = predict("every")
    prob_dict = predict(" ".join(subject))
    
    
    filtered_nouns = [x for x in nouns if x in prob_dict and x not in adjectives]
#     nouns = [x for x in nouns if x not in adjectives]

    sorted_nouns = sorted(filtered_nouns,key=lambda x : prob_dict[x],reverse=True)
    print("SOME SORTED NOUNS:",sorted_nouns[:50])
    
    filtered_adjs = [x for x in adjectives if x in vecs and x in frequencies and x in prob_dict]
    freq_sorted_adjs = sorted(filtered_adjs,key = lambda x : frequencies[x],reverse=True)

    quds = list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs["tree"]],axis=0),freq_sorted_adjs,vecs)))[0])

    quds=quds[:quds.index('unyielding')+1]
    
    run = DistRSAInference(
    subject=subject,predicate=word,
    possible_utterances=sorted_nouns[:sorted_nouns.index(word)+1],
#     quds=['strong','stable','wooden','connected','leafy','unyielding'],
    quds = quds,
#         ['unyielding']+list(list(zip(*visualize_cosine(np.mean([vecs['man'],vecs[word]],axis=0),freq_sorted_adjs,vecs)[:500:10]))[0]),

    # possible_utterances=nouns[:1000]+adjectives[:1000],
    object_name="animals_spec",
    mean_vecs=True,
    pca_remove_top_dims=False,
    sig1=1.0,sig2=tf.cast(scaling_factor*distance,dtype=tf.float32),
#         proposed_sig2*scaling_factor,
    qud_weight=0.0,freq_weight=10.0,
    categorical="categorical",
    vec_length=50,vec_type="glove.6B.",
    sample_number = 100,
    number_of_qud_dimensions=1,
    burn_in=70,
    seed=False,trivial_qud_prior=False,
    step_size=0.01,
    frequencies=prob_dict,
    qud_prior_weight=0.5
    )

    for i in range(1,2):
    for animal in ["lion"]:
        noncat2 = DistRSAInference(
            subject=['man'],predicate=animal,
            quds=animal_features,possible_utterances=[animal]+sorted_nouns[:1000],
            object_name="animals_cat",
            mean_vecs=True,
            pca_remove_top_dims=False,
            sig1=10.0,sig2=0.1,
            qud_weight=0.0,freq_weight=1.0,
            categorical="categorical",
            vec_length=50,vec_type="glove.6B.",
            sample_number = 2000,
            number_of_qud_dimensions=1,
            burn_in=1000,
            seed=False,trivial_qud_prior=False,
            step_size=0.1,
            frequencies=prob_dict
            )

        noncat2.compute_results(load=0,save=False)

