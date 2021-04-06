import logging
from typing import List
from pyrddl.pvariable import PVariable
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

    def _is_feature(self, name: str) -> bool:
        # todo n-arity
        return name in self.fluent_to_feature

    def _get_feature(self, name: str) -> str:
        # todo n-arity
        return self.fluent_to_feature[name]

    def _is_action(self, name: str, agent: Agent) -> bool:
        # todo n-arity
        return agent.name in self.actions and name in self.actions[agent.name]

    def _get_action(self, name: str, agent: Agent) -> str:
        # todo n-arity
        return self.actions[agent.name][name]

    def _is_constant(self, name: str) -> bool:
        return any(c[0] == name for c in self.constants.keys())

    def _get_constant_value(self, name: str) -> object:
        # todo n-arity
        return next(val for c, val in self.constants.items() if c[0] == name)

    def _is_enum(self, name: str) -> bool:
        for t, r in self.model.domain.types:
            if t == name:
                return True
        return False

    def _is_enum_type(self, name: str) -> bool:
        for _, r in self.model.domain.types:
            if name in r:
                return True
        return False

    def _get_enum_types(self, name: str) -> List[str] or None:
        for t, r in self.model.domain.types:
            if t == name:
                return r
        return None

    def _get_domain(self, t_range):
        # checks normal types
        if t_range == 'int':
            return int, 0.
        if t_range == 'bool':
            return bool, 0.
        if t_range == 'real':
            return float, 0.

        # checks enumerated / custom types
        domain = self._get_enum_types(t_range)
        if domain is not None:
            return list, domain

        raise ValueError(f'Could not get domain for range type: {t_range}!')

    def _create_world_agents(self, agent_name: str):
        # create world and agent #TODO read agent(s) name(s) from RDDL?
        logging.info('__________________________________________________')
        self.world = World()

        # create agent and set properties from instance
        agent = self.world.addAgent(agent_name)
        if hasattr(self.model.instance, 'horizon'):
            agent.setAttribute('horizon', self.model.instance.horizon)
        if hasattr(self.model.instance, 'discount'):
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

    def _create_feature(self, fluent: PVariable, agent: Agent) -> str:
        # todo n-arity
        f_name = f'{fluent.name}'
        domain = self._get_domain(fluent.range)

        # create feature
        f = self.world.defineState(agent.name, f_name, *domain)
        self.fluent_to_feature[fluent.name] = f

        # set to default value (if list assume first of list)
        lo = self.world.variables[f]['lo']
        def_val = fluent.default if fluent.default is not None else \
            lo if lo is not None else self.world.variables[f]['elements'][0]
        self.world.setFeature(f, def_val)

        logging.info(f'Created feature "{f}" from {fluent.fluent_type} "{fluent.name}" of type "{fluent.range}"')
        return f

    def _convert_variables(self, agent: Agent):
        # create variables from state fluents
        logging.info('__________________________________________________')
        self.fluent_to_feature = {}
        for sf in self.model.domain.state_fluents.values():
            # todo n-arity
            self._create_feature(sf, agent)

        # create variables from intermediate fluents
        for sf in self.model.domain.intermediate_fluents.values():
            # todo n-arity
            self._create_feature(sf, agent)

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
                continue
            f = self._get_feature(sf)
            self.world.setFeature(f, val)
            logging.info(f'Initialized feature "{f}" with value "{val}"')
