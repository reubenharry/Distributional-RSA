# generate stimuli
import glob
# import sys
import os
import configparser


def main():
    config = configparser.ConfigParser()
    config.read('../experiment.txt')

    fileDestination = '../static/scripts/'
    stimulusSource = config['PATHS']['resources_path'] + 'stimuli'
    exp_name = config['EXPERIMENT']['experiment_name']
    stimulusPath = '/static/resources/%s/stimuli/' % exp_name
    stimuli = ''

    def add_stimuli(varName, stimuli):
        stimuli += 'var ' + varName + ' = [ \n'

        for i, file in enumerate(glob.glob(stimulusSource + "/*.png")):
            stim_variables = os.path.splitext(
                os.path.basename(file))[0].split('-')
            stim = dict(zip(stim_variables[::2], stim_variables[1::2]))
            path = stimulusPath + os.path.basename(file)
            stimuli += "\t{ stimulus: '" + path + "'," + \
                       " exp: '" + stim['exp'] + "'," + \
                       " chain: '" + stim['chain'] + "'," + \
                       " generation: " + stim['gen'] + "},\n"

        stimuli = stimuli[:-2]
        stimuli += '\n];'
        return stimuli

    stimuli = add_stimuli('stimuli', stimuli)

    print(stimuli)
    # save in file
    f = open(fileDestination + 'stimuli.js', 'w+')
    f.write(stimuli)
    f.close()


# if __name__ == "__main__":
main()
