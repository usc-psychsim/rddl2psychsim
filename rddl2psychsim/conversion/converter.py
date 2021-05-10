import logging
from rddl2psychsim.conversion.constraints import _ConstraintsConverter
from rddl2psychsim.rddl import parse_rddl_file, parse_rddl

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class Converter(_ConstraintsConverter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_file(self, rddl_file: str, verbose=True) -> None:
        # parse RDDL file, set model
        self.model = parse_rddl_file(rddl_file, verbose)
        self._convert_rddl()
        logging.info('==================================================')
        logging.info(f'Done processing {rddl_file}!')

    def convert_str(self, rddl_str: str, verbose=True) -> None:
        # parse RDDL string, set model
        self.model = parse_rddl(rddl_str, verbose)
        self._convert_rddl()
        logging.info('==================================================')
        logging.info(f'Done processing RDDL string!')

    def _convert_rddl(self):
        logging.info('==================================================')
        logging.info(f'Converting RDDL domain "{self.model.domain.name}" '
                     f'using instance "{self.model.instance.name}" to PsychSim...')

        self._create_world_agents()
        self._convert_constants()
        self._convert_variables()
        self._convert_actions()
        self._initialize_variables()
        self._convert_reward_function()
        self._convert_dynamics()
        self._convert_state_action_constraints()
        self._parse_requirements()

    def get_players(self):
        return list(self.world.agents.keys())