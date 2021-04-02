import logging
from typing import List
from pyrddl.rddl import RDDL
from psychsim.agent import Agent
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class _ConverterBase(object):
    model: RDDL
    world: World

    def __init__(self):
        self.fluent_to_feature = {}
        self.constants = {}
        self.actions = {}

    def log_state(self, features: List[str] = None):
        for f in self.fluent_to_feature.values():
            if features is None or f in features:
                val = str(self.world.getFeature(f)).replace('\n', '\t')
                logging.info(f'{f}: {val}')

    def _is_feature(self, sf: str) -> bool:
        # todo n-arity
        return sf in self.fluent_to_feature

    def _get_feature(self, sf: str) -> str:
        # todo n-arity
        return self.fluent_to_feature[sf]

    def _is_action(self, af: str, agent: Agent) -> bool:
        # todo n-arity
        return agent.name in self.actions and af in self.actions[agent.name]

    def _get_action(self, af: str, agent: Agent) -> str:
        # todo n-arity
        return self.actions[agent.name][af]

    def _is_constant(self, nf: str) -> bool:
        return any(c[0] == nf for c in self.constants.keys())

    def _get_constant_value(self, nf: str) -> object:
        # todo n-arity
        return next(val for c, val in self.constants.items() if c[0] == nf)

    def _create_world_agents(self, agent_name: str):
        # create world and agent #TODO read agent(s) name(s) from RDDL?
        logging.info('__________________________________________________')
        self.world = World()

        # create agent and set properties from instance
        agent = self.world.addAgent(agent_name)
        agent.setAttribute('horizon', self.model.instance.horizon)
        agent.setAttribute('discount', self.model.instance.discount)

        # TODO other world and agent attributes
        agent.setAttribute('selection', 'random')

        model = agent.get_true_model()
        logging.info(f'Created agent {agent.name} with properties:')
        logging.info(f'\thorizon: {agent.getAttribute("horizon", model)}')
        logging.info(f'\tdiscount: {agent.getAttribute("discount", model)}')
        return agent

    def _convert_constants(self):
        # first set value of non-fluents from definition
        logging.info('__________________________________________________')
        self.constants = {}
        if hasattr(self.model.non_fluents, 'init_non_fluent'):
            for nf, val in self.model.non_fluents.init_non_fluent:
                self.constants[nf] = val
                logging.info(f'Initialized constant "{nf}" with value "{val}"')

        # try to initialize non-fluents from definition's default value
        for nf in self.model.domain.non_fluents.values():
            if nf.arity != 0:  # can't initialize parameterizable constants
                continue
            nf_name = (nf.name, None)
            if nf_name not in self.constants:  # non-fluent definition on file takes precedence
                self.constants[nf_name] = nf.default
                logging.info(f'Initialized constant "{nf_name}" with default value "{nf.default}"')

        logging.info(f'Total {len(self.constants)} constants initialized')

    def _convert_variables(self, agent: Agent):
        # create variables from state fluents
        logging.info('__________________________________________________')
        self.fluent_to_feature = {}
        for sf in self.model.domain.state_fluents.values():
            # todo n-arity
            f_name = f'{sf.name}'
            f = self.world.defineState(agent.name, f_name, type(sf.default))
            self.world.setFeature(f, sf.default)
            self.fluent_to_feature[sf.name] = f
            logging.info(f'Created feature "{f}" from state fluent "{sf.name}"')

        logging.info(f'Total {len(self.fluent_to_feature)} features created')

    def _convert_actions(self, agent: Agent):
        # create actions for agent
        logging.info('__________________________________________________')
        self.actions = {agent.name: {}}
        for act in self.model.domain.action_fluents.values():
            action = agent.addAction({'verb': act.name})
            self.actions[agent.name][act.name] = action
            logging.info(f'Created action "{action}" from action fluent: {act}')

        logging.info(f'Total {len(self.actions[agent.name])} actions created for agent "{agent.name}"')

    def _initialize_variables(self):

        # initialize variables from instance def
        logging.info('__________________________________________________')
        for sf, val in self.model.instance.init_state:
            # todo n-arity
            sf = sf[0]
            if not self._is_feature(sf):
                logging.info(f'Could not find feature corresponding to fluent "{sf}", skipping')
            f = self._get_feature(sf)
            self.world.setFeature(f, val)
            logging.info(f'Initialized feature "{f}" with value "{val}"')
