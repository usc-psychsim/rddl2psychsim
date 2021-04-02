import logging
from rddl2psychsim.conversion.pwl import _ConverterPWLBase
from rddl2psychsim.rddl import parse_rddl

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class Converter(_ConverterPWLBase):

    def __init__(self):
        super().__init__()

    def convert(self, rddl_file: str, agent_name='Agent', verbose=True) -> None:
        # parse RDDL file, set model
        self.model = parse_rddl(rddl_file, verbose)

        logging.info('==================================================')
        logging.info(f'Converting RDDL domain "{self.model.domain.name}" '
                     f'using instance "{self.model.instance.name}" to PsychSim...')

        # TODO maybe read agent(s) names from rddl file? How would we separate stuff? how about models?
        agent = self._create_world_agents(agent_name)

        self._convert_constants()
        self._convert_variables(agent)
        self._convert_actions(agent)
        self._convert_reward_function(agent)
        self._convert_dynamics(agent)
        self._initialize_variables()

        self.world.setOrder([{agent.name}])

        logging.info('==================================================')
        logging.info(f'Done processing {rddl_file}!')
