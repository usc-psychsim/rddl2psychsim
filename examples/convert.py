import os
import argparse
import logging
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__desc__ = 'Converts an RDDL file to PsychSim and runs a simulation for some timesteps.'

MAX_STEPS = 50
RDDL_FILE = 'examples/files/dbn_prop.rddl'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__desc__)
    parser.add_argument('--input', '-i', type=str, default=RDDL_FILE, help='RDDL file to be converted to PsychSim.')
    parser.add_argument('--steps', '-s', type=str, default=MAX_STEPS, help='Number of steps to run the simulation.')
    args = parser.parse_args()

    # prepare log
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    # parse and convert RDDL file
    if not os.path.isfile(args.input):
        raise ValueError(f'Could not find RDDL file: {args.input}')
    conv = Converter()
    conv.convert(args.input, verbose=False)

    logging.info('')
    logging.info('==================================================')
    logging.info('Starting simulation...')
    conv.log_state()
    for i in range(args.steps):
        logging.info('__________________________________________________')
        logging.info(f'Step {i}:')
        conv.world.step()
        conv.log_state()

    logging.info('==================================================')
    logging.info('Done!')
