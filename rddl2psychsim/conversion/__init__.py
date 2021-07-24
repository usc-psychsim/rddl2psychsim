import itertools
import logging
import re
import numpy as np
import scipy.stats as stats
from collections import OrderedDict
from typing import List, Tuple, Set, Dict
from pyrddl.pvariable import PVariable
from pyrddl.rddl import RDDL
from psychsim.action import ActionSet
from psychsim.agent import Agent
from psychsim.pwl import actionKey, WORLD
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

# default discretization for distributions
NORMAL_STDS = 3
NORMAL_BINS = 7
POISSON_EXP_RATE = 10

MAX_LEVEL = int(1e10)


class _ConverterBase(object):
    model: RDDL
    world: World

    turn_order: List[Set[str]]

    def __init__(self, normal_stds=NORMAL_STDS, normal_bins=NORMAL_BINS, poisson_exp_rate=POISSON_EXP_RATE):
        self.features: Dict[str, str] = {}
        self.constants: Dict[str, int or float or bool or str] = {}
        self.actions: Dict[str, Dict[str, ActionSet]] = OrderedDict()  # order of agents is as given in RDDL instance
        self._fluent_param_types: Dict[str, List[str]] = {}
        self._fluent_levels: Dict[str, int] = {}

        # set distribution discretization params
        self._normal_bins = np.linspace(-normal_stds, normal_stds, normal_bins).tolist()  # gets sample value centers
        self._normal_probs = stats.norm.pdf(self._normal_bins)  # gets corresponding sample probabilities
        self._normal_probs = (self._normal_probs / self._normal_probs.sum()).tolist()  # normalize to sum 1
        self._poisson_exp_rate = poisson_exp_rate

    def log_state(self, features: List[str] = None, log_actions: bool = False) -> None:
        """
        Logs (INFO level) the current state of the PsychSim world.
        Only prints features that were converted from RDDL.
        :param List[str] features: the features whose current value are to be printed. `None` will print all
        :param bool log_actions: whether to also log agents' actions
        features on record.
        """
        for f in self.features.values():
            if features is None or f in features:
                val = str(self.world.getFeature(f)).replace('\n', '\t')
                logging.info(f'{f}: {val}')

        for ag_name in self.actions.keys():
            if log_actions:
                f = actionKey(ag_name)
                val = str(self.world.getFeature(f)).replace('\n', '\t')
                logging.info(f'{f}: {val}')

    def get_agents(self):
        return list(self.world.agents.keys())

    @staticmethod
    def get_feature_name(f: Tuple) -> str:
        """
        Gets a PsychSim feature identifier name for the given (possibly parameterized) fluent.
        :param Tuple f: the (possibly parameterized) fluent, e.g., `('p', None)` or `('p', x1, y1)`.
        :rtype: str
        :return: the identifier string for the fluent.
        """
        if isinstance(f, tuple):
            f = tuple(n for n in f if n is not None)
            if len(f) == 1:
                f = f[0]
            return re.sub(r'\'|"|@', '', str(f))
        return str(f)

    def _is_feature(self, name: Tuple) -> bool:
        return self.get_feature_name(name) in self.features

    def _get_feature(self, name: Tuple) -> str:
        return self.features[self.get_feature_name(name)]

    def _is_action(self, name: Tuple, agent: Agent) -> bool:
        return agent.name in self.actions and self.get_feature_name(name) in self.actions[agent.name]

    def _get_action(self, name: Tuple, agent: Agent) -> ActionSet:
        return self.actions[agent.name][self.get_feature_name(name)]

    def _is_constant(self, name: Tuple) -> bool:
        return self.get_feature_name(name) in self.constants

    def _is_constant_value(self, val: str) -> bool:
        return val in self.constants.values()

    def _get_constant_value(self, name: Tuple) -> object:
        return self.constants[self.get_feature_name(name)]

    def _get_entity_name(self, name: Tuple) -> Tuple[str, Tuple]:
        name = list(name)
        # searches for agent name in (possibly parameterized) variable's name
        for n in name:
            if n in self.world.agents:
                name.remove(n)
                return n, tuple(name)
        return WORLD, tuple(name)  # defaults to world

    def _is_enum(self, name: str) -> bool:
        for t_name, t_vals in self.model.domain.types:
            if t_name == name and isinstance(t_vals, list) and len(t_vals) > 0:
                return True
        return False

    def _is_enum_value(self, val: str) -> bool:
        for t_name, t_vals in self.model.domain.types:
            if self._is_enum(t_name) and (val in t_vals or '@' + val in t_vals):
                return True
        return False

    def _get_enum_values(self, name: str) -> List[str] or None:
        if not self._is_enum(name):
            return None
        for t_name, t_vals in self.model.domain.types:
            if t_name == name:
                return [_.replace('@', '') for _ in t_vals]  # strip "@" character from enum values
        return None

    def _get_domain(self, t_range: str):
        # checks normal types
        if t_range == 'int':
            return int, 0.
        if t_range == 'bool':
            return bool, 0.
        if t_range == 'real':
            return float, 0.

        # checks enumerated (domain-level constant) types
        domain = self._get_enum_values(t_range)
        if domain is not None:
            return list, domain

        # checks object (instance-level constant) types
        try:
            domain = self._get_param_values(t_range)
            if domain is not None:
                return list, domain
        except ValueError:
            pass

        raise ValueError(f'Could not get domain for range type: {t_range}!')

    def _get_param_types(self, name: str) -> List[str]:
        assert name in self._fluent_param_types, \
            f'Could not get param types for fluent: {name}, feature not registered!'
        return self._fluent_param_types[name]

    def _get_param_values(self, param_type: str) -> List[str]:
        if self._is_enum(param_type):  # check enum type
            return self._get_enum_values(param_type)
        for p_type, p_vals in self.model.non_fluents.objects:  # check object instance type
            if p_type == param_type:
                return p_vals
        raise ValueError(f'Could not get values for param type: {param_type}!')

    def _get_all_param_combs(self, param_types: List[str]) -> List[Tuple]:
        param_vals = [self._get_param_values(p_type) for p_type in param_types]
        return list(itertools.product(*param_vals))

    def _create_world_agents(self) -> None:
        # create world
        logging.info('__________________________________________________')
        self.world = World()

        # create agents from RDDL non-fluent definition, special object named "agent"
        for p_type, p_vals in self.model.non_fluents.objects:
            if p_type == 'agent':
                for ag_name in p_vals:
                    self.world.addAgent(ag_name)
                break
        if len(self.world.agents) == 0:
            self.world.addAgent('AGENT')  # create default agent if no agents defined

        # set agents' properties from instance
        for agent in self.world.agents.values():
            if hasattr(self.model.instance, 'horizon'):
                agent.setAttribute('horizon', self.model.instance.horizon)
            if hasattr(self.model.instance, 'discount'):
                agent.setAttribute('discount', self.model.instance.discount)

            # TODO other world and agent attributes?
            agent.setAttribute('selection', 'random')

            model = agent.get_true_model()
            logging.info(f'Created agent "{agent.name}" with properties:')
            logging.info(f'\thorizon: {agent.getAttribute("horizon", model)}')
            logging.info(f'\tdiscount: {agent.getAttribute("discount", model)}')

    def _convert_constants(self):
        logging.info('__________________________________________________')
        self.constants = {}

        # creates constants from object values (the values might be used in expressions)
        for obj_name, obj_values in self.model.non_fluents.objects:
            self.constants.update({obj_val: obj_val for obj_val in obj_values})
            logging.info(f'Added {len(obj_values)} constant values for object type "{obj_name}"')

        # try to initialize non-fluents from definition's default value
        for nf in self.model.domain.non_fluents.values():
            if nf.arity > 0:
                # gets all parameter combinations
                param_vals = self._get_all_param_combs(nf.param_types)
                nf_combs = [(nf.name, *p_vals) for p_vals in param_vals]
            else:
                nf_combs = [(nf.name, None)]  # not-parameterized constant
            for nf_name in nf_combs:
                nf_name = self.get_feature_name(nf_name)
                def_val = nf.default if nf.default not in {None, 'none', 'null', 'None', 'Null'} else \
                    self._get_domain(nf.range)[1][0]
                if isinstance(def_val, str):
                    def_val = def_val.replace('@', '')  # just in case it's an enum value
                self.constants[nf_name] = def_val
                logging.info(f'Initialized constant "{nf_name}" with default value "{def_val}"')

        # then set value of non-fluents from initialization definition
        if hasattr(self.model.non_fluents, 'init_non_fluent'):
            for nf, def_val in self.model.non_fluents.init_non_fluent:
                nf_name = nf if nf[1] is None else (nf[0], *nf[1])
                nf_name = self.get_feature_name(nf_name)
                if nf_name not in self.constants:
                    raise ValueError(f'Trying to initialize non-existing non-fluent: {nf_name}!')
                def_val = def_val if def_val not in {None, 'none', 'null', 'None', 'Null'} else \
                    self._get_domain(nf.range)[1][0]
                if isinstance(def_val, str):
                    def_val = def_val.replace('@', '')  # just in case it's an enum value
                self.constants[nf_name] = def_val
                logging.info(f'Initialized constant "{nf_name}" in non-fluents to "{def_val}"')

        logging.info(f'Total {len(self.constants)} constants initialized')

    def _convert_variables(self):
        # create features from state fluents
        logging.info('__________________________________________________')
        self.features = {}
        for sf in self.model.domain.state_fluents.values():
            self._create_features(sf, '' if len(self.model.domain.observ_fluents) == 0 else '__')

        # create features from intermediate fluents
        for sf in self.model.domain.intermediate_fluents.values():
            self._create_features(sf, '_')

        # create features from observable fluents
        for sf in self.model.domain.observ_fluents.values():
            self._create_features(sf)

        logging.info(f'Total {len(self.features)} features created')

    def _create_features(self, fluent: PVariable, prefix: str = '') -> List[str]:
        # registers types of parameters for this type of feature
        self._fluent_param_types[fluent.name] = fluent.param_types
        self._fluent_levels[fluent.name] = fluent.level if fluent.fluent_type == 'interm-fluent' else \
            -1 if fluent.fluent_type == 'state-fluent' else MAX_LEVEL

        # to whom should this feature be associated, agent or world?
        if fluent.arity > 0:
            # gets all parameter combinations
            param_vals = self._get_all_param_combs(fluent.param_types)
            f_combs = [(fluent.name, *p_vals) for p_vals in param_vals]
        else:
            f_combs = [(fluent.name, None)]  # not-parameterized feature

        # create and register features
        feats = []
        domain = self._get_domain(fluent.range)
        for f_name in f_combs:
            entity, feat_name = self._get_entity_name(f_name)  # tries to identify agent from fluent param comb name
            f = self.world.defineState(entity, prefix + self.get_feature_name(feat_name), *domain)
            f_name = self.get_feature_name(f_name)  # keep feature's original name for transparent later referencing
            self.features[f_name] = f

            # set to default value (if list assume first of list)
            lo = self.world.variables[f]['lo']
            def_val = fluent.default if fluent.default not in {None, 'none', 'null', 'None', 'Null'} else \
                lo if lo is not None else self.world.variables[f]['elements'][0]
            if isinstance(def_val, str):
                def_val = def_val.replace('@', '')  # just in case it's an enum value
            self.world.setFeature(f, def_val)

            logging.info(f'Created feature "{f}" from {fluent.fluent_type} "{fluent.name}" of type "{fluent.range}"')
            feats.append(f)
        return feats

    def _convert_actions(self):
        # create actions for agents (assume homogeneous agents) TODO maybe put constraints in RDDL for diff agents?
        logging.info('__________________________________________________')
        self.actions = {agent.name: {} for agent in self.world.agents.values()}
        for act_fluent in self.model.domain.action_fluents.values():
            self._create_actions(act_fluent)
        logging.info(f'Total {sum(len(actions) for actions in self.actions.values())} actions created')

    def _create_actions(self, fluent: PVariable) -> List[ActionSet]:
        self._fluent_levels[fluent.name] = -MAX_LEVEL  # for dynamics, actions always come first

        if fluent.arity > 0:
            # gets all parameter combinations
            param_vals = self._get_all_param_combs(fluent.param_types)
            act_combs = [(fluent.name, *p_vals) for p_vals in param_vals]
        else:
            act_combs = [(fluent.name, None)]  # not-parameterized action

        # create action for each param combination
        actions = []
        for act_name in act_combs:
            entity, action_name = self._get_entity_name(act_name)  # tries to identify agent from fluent param comb name
            if entity == WORLD:
                agents = self.world.agents.values()  # create action for all agents
            else:
                agents = [self.world.agents[entity]]  # create action just for this agent

            for agent in agents:
                # keep feature's original name for transparent later referencing
                act_name = self.get_feature_name(act_name)
                action = agent.addAction({'verb': self.get_feature_name(action_name)})
                actions.append(action)
                self.actions[agent.name][act_name] = action
                logging.info(f'Created action "{action}" for agent "{agent.name}" from action fluent: {fluent}')
        return actions

    def _initialize_variables(self):
        # initialize variables from instance def
        logging.info('__________________________________________________')
        if not hasattr(self.model.instance, 'init_state'):
            logging.info('Domain instance does not have "init_state" section, skipping feature initialization')
            return  # no initial state definitions

        for sf, val in self.model.instance.init_state:
            f_name = sf if sf[1] is None else (sf[0], *sf[1])
            if any(self._is_action(f_name, agent) for agent in self.world.agents.values()):
                continue  # skip action initialization
            assert self._is_feature(f_name), f'Could not find feature "{f_name}" corresponding to fluent "{sf}"!'
            f = self._get_feature(f_name)
            if isinstance(val, str):
                val = val.replace('@', '')  # just in case it's an enum value
            self.world.setFeature(f, val)
            logging.info(f'Initialized feature "{f}" to "{val}"')

    def _parse_requirements(self):

        logging.info('__________________________________________________')
        logging.info('Parsing requirements...')
        agents = self.world.agents
        requirements = self.model.domain.requirements

        # check concurrent multiagent actions, agent order is assumed by RDDL instance definition
        if len(agents) > 1 and requirements is not None and 'concurrent' in requirements:
            if hasattr(self.model.instance, 'max_nondef_actions'):
                # creates groups of agents that act in parallel according to "max_nondef_actions" param
                num_parallel = self.model.instance.max_nondef_actions
                ag_list = list(self.actions.keys())
                self.turn_order = []
                for i in range(0, len(agents), num_parallel):
                    self.turn_order.append(set(ag_list[i:min(i + num_parallel, len(agents))]))
            else:
                self.turn_order = [set(agents.keys())]  # assumes all agents act in parallel
        else:
            self.turn_order = [{ag} for ag in agents.keys()]  # assumes all agents act sequentially
        self.world.setOrder(self.turn_order)

        # sets omega to observe only observ-fluents (ignore interm and state-fluents)
        # TODO assume homogeneous agents (partial observability affects all agents the same)
        if requirements is not None and 'partially-observed' in requirements:
            for agent in agents.values():
                observable = [actionKey(agent.name)]  # todo what do we need to put here?
                for sf in self.model.domain.observ_fluents.values():
                    if self._is_feature(sf.name):
                        observable.append(self._get_feature(sf.name))
                agent.omega = observable
                logging.info(f'Setting partial observability for agent "{agent.name}", omega={agent.omega}')
