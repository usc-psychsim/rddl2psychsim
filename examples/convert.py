import os
import argparse
import logging
from tqdm import tqdm
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__desc__ = 'Converts an RDDL file to PsychSim and runs a simulation for some timesteps.'

MAX_STEPS = 50
THRESHOLD = 0
RDDL_FILE = 'examples/domains/dbn_prop.rddl'


def _log_agent_reward(ag_name):
    if '__decision__' not in debug[ag_name]:
        return
    true_model = conv.world.agents[ag_name].get_true_model()
    action = debug[ag_name]['__decision__'][true_model]['action']
    rwd = debug[ag_name]['__decision__'][true_model]['V'][action]['__ER__']
    rwd = None if len(rwd) == 0 else rwd[0]
    logging.info(f'{ag_name}\'s reward: {rwd}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__desc__)
    parser.add_argument('--input', '-i', type=str, default=RDDL_FILE, help='RDDL file to be converted to PsychSim.')
    parser.add_argument('--steps', '-s', type=int, default=MAX_STEPS, help='Number of steps to run the simulation.')
    parser.add_argument('--threshold', '-t', type=float, default=THRESHOLD,
                        help='Stochastic outcomes with a likelihood below this threshold are pruned.')
    parser.add_argument('--select', action='store_true',
                        help='Whether to select an outcome if dynamics are stochastic.')
    parser.add_argument('--log-actions', action='store_true',
                        help='Whether to log agents\' actions in addition to current state.')
    parser.add_argument('--log-rewards', action='store_true',
                        help='Whether to log agents\' rewards wrt chosen actions in addition to current state.')
    args = parser.parse_args()

    # prepare log to screen
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    # parse and convert RDDL file
    if not os.path.isfile(args.input):
        raise ValueError(f'Could not find RDDL file: {args.input}')
    conv = Converter()
    conv.convert_file(args.input, verbose=True)

    logging.info('')
    logging.info('==================================================')
    logging.info('Starting simulation...')
    conv.log_state()
    for i in tqdm(range(args.steps), ):
        logging.info('\n__________________________________________________')
        logging.info(f'Step {i}:')
        debug = {ag_name: {} for ag_name in conv.actions.keys()} if args.log_rewards else set()
        conv.world.step(debug=debug, threshold=args.threshold, select=args.select)
        conv.log_state(log_actions=args.log_actions)
        if args.log_rewards:
            for ag_name in conv.actions.keys():
                _log_agent_reward(ag_name)
        conv.verify_constraints()

    logging.info('==================================================')
    logging.info('Done!')
