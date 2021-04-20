import os
import argparse
import logging
import numpy as np
from tqdm import tqdm
from psychsim.pwl import actionKey, WORLD
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__desc__ = 'Converts the Game-of-Life RDDL file to PsychSim and runs a simulation for some timesteps.'

MAX_STEPS = 50
THRESHOLD = 0
RDDL_FILE = 'examples/domains/game_of_life_stoch.rddl'


def print_state():
    objs = dict(conv.model.non_fluents.objects)
    x_objs = objs['x_pos']
    y_objs = objs['y_pos']
    state = np.zeros((len(x_objs), len(y_objs)), dtype=int)
    for x in range(len(x_objs)):
        x_obj = x_objs[x]
        for y in range(len(y_objs)):
            y_obj = y_objs[y]
            f_name = Converter.get_fluent_name(('alive', x_obj, y_obj))
            state[x, y] = int(conv.world.getState(WORLD, f_name, unique=True))
    logging.info(f'Action: {conv.world.getFeature(actionKey(next(iter(conv.world.agents.keys()))), unique=True)}')
    logging.info(f'\n{state}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__desc__)
    parser.add_argument('--steps', '-s', type=str, default=MAX_STEPS, help='Number of steps to run the simulation.')
    args = parser.parse_args()

    # prepare log t screen
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    # parse and convert RDDL file
    if not os.path.isfile(RDDL_FILE):
        raise ValueError(f'Could not find RDDL file: {RDDL_FILE}')
    conv = Converter()
    conv.convert_file(RDDL_FILE, verbose=True)

    logging.info('')
    logging.info('==================================================')
    logging.info('Starting simulation...')
    print_state()
    for i in tqdm(range(args.steps), ):
        logging.info('\n__________________________________________________')
        logging.info(f'Step {i}:')
        conv.world.step(select=True)
        print_state()

    logging.info('==================================================')
    logging.info('Done!')
