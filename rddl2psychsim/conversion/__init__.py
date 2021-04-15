import itertools
import logging
import numpy as np
import scipy.stats as stats
from typing import List, Tuple
from pyrddl.pvariable import PVariable
from pyrddl.rddl import RDDL
from psychsim.agent import Agent
from psychsim.pwl import actionKey
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

# default discretization for distributions
NORMAL_STDS = 3
NORMAL_BINS = 7
POISSON_EXP_RATE = 10


class _ConverterBase(object):
    model: RDDL
    world: World

    _normal_bins: List[float]
    _normal_probs: List[float]
    _poisson_exp_rate: int
    _poisson_bins: List[int]
    _poisson_probs: List[float]

    def __init__(self):
        self.fluent_to_feature = {}
        self.constants = {}
        self.actions = {}

    def log_state(self, features: List[str] = None) -> None:
        """
        Logs (INFO level) the current state of the PsychSim world.
        Only prints features that were converted from RDDL.
        :param List[str] features: the features whose current value are to be printed. `None` will print all
        features on record.
        """
        for f in self.fluent_to_feature.values():
            if features is None or f in features:
                val = str(self.world.getFeature(f)).replace('\n', '\t')
                logging.info(f'{f}: {val}')

    @staticmethod
    def get_fluent_name(f: Tuple) -> str:
        """
        Gets an identifier name for the given (possibly parameterized) fluent.
        :param Tuple f: the (possibly parameterized) fluent, e.g., `('p', None)` or `('p', x1, y1)`.
        :rtype: str
        :return: the identifier string for the fluent.
        """
        if isinstance(f, tuple):
            f = tuple(n for n in f if n is not None)
            if len(f) == 1:
                f = f[0]
            return str(f).replace('\'', '').replace('"', '')
        return str(f)

    def _is_feature(self, name: Tuple) -> bool:
        return self.get_fluent_name(name) in self.fluent_to_feature

    def _get_feature(self, name: Tuple) -> str:
        return self.fluent_to_feature[self.get_fluent_name(name)]

    def _is_action(self, name: Tuple, agent: Agent) -> bool:
        return agent.name in self.actions and self.get_fluent_name(name) in self.actions[agent.name]

    def _get_action(self, name: Tuple, agent: Agent) -> str:
        return self.actions[agent.name][self.get_fluent_name(name)]

    def _is_constant(self, name: Tuple) -> bool:
        return self.get_fluent_name(name) in self.constants

    def _get_constant_value(self, name: Tuple) -> object:
        return self.constants[self.get_fluent_name(name)]

    def _is_enum(self, name: str) -> bool:
        for t, _ in self.model.domain.types:
            if t == name:
                return True
        return False

    def _is_enum_type(self, name: str) -> bool:
        for _, r in self.model.domain.types:
            if name in r or '@' + name in r:
                return True
        return False

    def _get_enum_types(self, name: str) -> List[str] or None:
        for t, r in self.model.domain.types:
            if t == name:
                return [_.replace('@', '') for _ in r]  # strip "@" character from enum values
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

    def _get_param_values(self, param_type: str) -> List[str]:
        for p_type, p_vals in self.model.non_fluents.objects:
            if p_type == param_type:
                return p_vals
        raise ValueError(f'Could not get values for param type: {param_type}!')

    def _get_all_param_combs(self, param_types: List[str]) -> List[Tuple]:
        param_vals = [self._get_param_values(p_type) for p_type in param_types]
        return list(itertools.product(*param_vals))

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
        logging.info(f'Created agent "{agent.name}" with properties:')
        logging.info(f'\thorizon: {agent.getAttribute("horizon", model)}')
        logging.info(f'\tdiscount: {agent.getAttribute("discount", model)}')
        return agent

    def _convert_constants(self):
        # first try to initialize non-fluents from definition's default value
        logging.info('__________________________________________________')
        self.constants = {}
        for nf in self.model.domain.non_fluents.values():
            if nf.arity > 0:
                # gets all parameter combinations
                param_vals = self._get_all_param_combs(nf.param_types)
                nf_combs = [(nf.name, *p_vals) for p_vals in param_vals]
            else:
                nf_combs = [(nf.name, None)]  # not-parameterized constant
            for nf_name in nf_combs:
                nf_name = self.get_fluent_name(nf_name)
                self.constants[nf_name] = nf.default
                logging.info(f'Initialized constant "{nf_name}" with default value "{nf.default}"')

        # then set value of non-fluents from initialization definition
        if hasattr(self.model.non_fluents, 'init_non_fluent'):
            for nf, val in self.model.non_fluents.init_non_fluent:
                nf_name = nf if nf[1] is None else (nf[0], *nf[1])
                nf_name = self.get_fluent_name(nf_name)
                if nf_name not in self.constants:
                    raise ValueError(f'Trying to initialize non-existing non-fluent: {nf_name}!')
                self.constants[nf_name] = val
                logging.info(f'Initialized constant "{nf}" with value "{val}"')

        logging.info(f'Total {len(self.constants)} constants initialized')

    def _create_features(self, fluent: PVariable, agent: Agent) -> List[str]:
        if fluent.arity > 0:
            # gets all parameter combinations
            param_vals = self._get_all_param_combs(fluent.param_types)
            f_combs = [(fluent.name, *p_vals) for p_vals in param_vals]
        else:
            f_combs = [(fluent.name, None)]  # not-parameterized constant

        # create and register features
        feats = []
        domain = self._get_domain(fluent.range)
        for f_name in f_combs:
            f_name = self.get_fluent_name(f_name)
            f = self.world.defineState(agent.name, f_name, *domain)
            self.fluent_to_feature[f_name] = f

            # set to default value (if list assume first of list)
            lo = self.world.variables[f]['lo']
            def_val = fluent.default if fluent.default is not None else \
                lo if lo is not None else self.world.variables[f]['elements'][0]
            if isinstance(def_val, str):
                def_val = def_val.replace('@', '')  # just in case it's an enum value
            self.world.setFeature(f, def_val)

            logging.info(f'Created feature "{f}" from {fluent.fluent_type} "{fluent.name}" of type "{fluent.range}"')
            feats.append(f)
        return feats

    def _convert_variables(self, agent: Agent):
        # create variables from state fluents
        logging.info('__________________________________________________')
        self.fluent_to_feature = {}
        for sf in self.model.domain.state_fluents.values():
            self._create_features(sf, agent)

        # create variables from intermediate fluents
        for sf in self.model.domain.intermediate_fluents.values():
            self._create_features(sf, agent)

        # create variables from non-observable fluents
        for sf in self.model.domain.observ_fluents.values():
            self._create_features(sf, agent)

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

    def _initialize_variables(self, agent: Agent):
        # initialize variables from instance def
        logging.info('__________________________________________________')
        for sf, val in self.model.instance.init_state:
            f_name = sf if sf[1] is None else (sf[0], *sf[1])
            if self._is_action(f_name, agent):
                continue  # skip action initialization
            assert self._is_feature(f_name), f'Could not find feature "{f_name}" corresponding to fluent "{sf}"!'
            f = self._get_feature(f_name)
            if isinstance(val, str):
                val = val.replace('@', '')  # just in case it's an enum value
            self.world.setFeature(f, val)
            logging.info(f'Initialized feature "{f}" with value "{val}"')

    def _parse_requirements_pre(self, agent: Agent):
        logging.info('__________________________________________________')

        # get Normal distribution discretization params
        normal_stds_req = [] if self.model.domain.requirements is None else \
            [req for req in self.model.domain.requirements if 'normal_stds' in req]
        if len(normal_stds_req) > 0:
            normal_stds = float(normal_stds_req[0].replace('normal_stds', ''))
        else:
            normal_stds = NORMAL_STDS
        logging.info(f'Using {normal_stds} standard devs to define the range of Normal distributions')

        normal_bins_req = [] if self.model.domain.requirements is None else \
            [req for req in self.model.domain.requirements if 'normal_bins' in req]
        if len(normal_bins_req) > 0:
            normal_bins = int(normal_bins_req[0].replace('normal_bins', ''))
        else:
            normal_bins = NORMAL_BINS
        logging.info(f'Using {normal_bins} discrete bins to define the values of Normal distributions')

        self._normal_bins = np.linspace(-normal_stds, normal_stds, normal_bins).tolist()  # gets sample value centers
        self._normal_probs = stats.norm.pdf(self._normal_bins)  # gets corresponding sample probabilities
        self._normal_probs = (self._normal_probs / self._normal_probs.sum()).tolist()  # normalize to sum 1

        # get Poisson distribution discretization params
        poisson_exp_rate_req = [] if self.model.domain.requirements is None else \
            [req for req in self.model.domain.requirements if 'poisson_exp_rate' in req]
        if len(poisson_exp_rate_req) > 0:
            self._poisson_exp_rate = int(poisson_exp_rate_req[0].replace('poisson_exp_rate', ''))
        else:
            self._poisson_exp_rate = POISSON_EXP_RATE
        logging.info(f'Using {self._poisson_exp_rate} as the expected rate of Poisson distributions')

    def _parse_requirements_post(self, agent: Agent):
        if self.model.domain.requirements is None or len(self.model.domain.requirements) == 0:
            return

        logging.info('__________________________________________________')

        # sets omega to observe only observ-fluents (ignore interm and state-fluents)
        if 'partially-observed' in self.model.domain.requirements:
            observable = [actionKey(agent.name)]  # todo what do we need to put here?
            for sf in self.model.domain.observ_fluents.values():
                if self._is_feature(sf.name):
                    observable.append(self._get_feature(sf.name))
            agent.omega = observable
            logging.info(f'Setting partial observability for agent "{agent.name}", omega={agent.omega}')
