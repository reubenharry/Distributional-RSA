# load specified version of experiment
import sys
import configparser
from shutil import copyfile


def main():
    # accept optional path
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        print("Need to pass argument to specify experiment version!")
        exit()

    # TODO:
    # read in exp-config
    # go into psiturk-config and exp.js and replace number of chains
    # number of learning blocks etc. here
    # whether or not to give visual feedback
    #
    # in the ad file, replace amout of bonus
    # complete_bonus
    # performance_bonus
    # total_pay
    #

    # Setting the main experiment file (exp.js)
    resources_path = '../static/resources/' + experiment_name + '/'

    exp_js_destination = '../static/scripts/'
    exp_js_filename = 'exp.js'

    comment = '// the main experimental script is now loaded.'
    fileStr = "$.getScript('%s%s', function(){\n\t%s\n});" % (
        resources_path, exp_js_filename, comment)

    f = open(exp_js_destination + exp_js_filename, 'w+')
    f.write(fileStr)
    f.close()

    print("Now using %s as main experiental file."
          % (experiment_name + '/' + exp_js_filename))

    # Writing the global config file (experiment.txt)
    config = configparser.ConfigParser()
    config['PATHS'] = {
        'experiment_name': experiment_name,
        'resources_path': resources_path,
        'signal_path': resources_path+'signals',
        'referent_path': resources_path+'referents'
    }
    config['EXPERIMENT'] = {
        'experiment_name': experiment_name,
        'feedback': 'yes'
    }

    # TODO: make backup of experiment file
    with open('../experiment.txt', 'w') as configfile:
        config.write(configfile)

    # Try - except: ad file missing!!!
    # overwrite the ad file
    copyfile(resources_path + 'ad.html', '../templates/ad.html')
    copyfile(resources_path + 'psiturk-config.txt', '../config.txt')


if __name__ == "__main__":
    main()
