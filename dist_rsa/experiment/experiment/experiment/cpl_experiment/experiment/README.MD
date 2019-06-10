[CONFIGURE THE EXPERIMENT]

# ad.html
Specify experiment description and bonuses

# psiturk-config.html
The psiturk config file

# exp-config
Currently unused



[MOUNT THE EXPERIMENT]

cd code
# select an experiment from /resources/{folder_name}
python load_experiment.py {folder_name}
python load_stimuli.py {generation}

# only for color patch referents (maybe make this file experiment specific):
python generate_referents.py



[RUNNING EXPERIMENTS]

psiturk
hit create {number of HITs}
{amount of base pay}
{time to work on the HIT}


# automatically pulls most recent generation from database
# and compiles visualizations
python analyze_results.py

[...]

# inspect and choose subjects to advance to next generation

