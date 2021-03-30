import logging
from typing import Dict

from pyrddl.rddl import RDDL
from pyrddl.expr import Expression

from psychsim.agent import Agent
from psychsim.world import World
from psychsim_rddl.rddl import _parse_rddl
from psychsim.pwl import KeyedVector, CONSTANT, rewardKey, makeTree, makeFuture, KeyedMatrix

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class Converter(object):

    def __init__(self):
        self.model: RDDL = None
        self.world: World = None
        self.fluent_to_feature = {}
        self.constants = {}
        self.actions = {}

    def convert(self, rddl_file: str, agent_name='Agent', verbose=True) -> None:
        # parse RDDL file
        self.model = _parse_rddl(rddl_file, verbose)
        domain = self.model.domain

        # create world and agent #TODO read agent(s) name(s) from RDDL?
        logging.info('__________________________________________________')
        self.world = World()
        agent = self.world.addAgent(agent_name)
        logging.info(f'Created agent {agent.name}')

        # first set value of non-fluents from definition
        logging.info('__________________________________________________')
        self.constants = {}
        if hasattr(self.model.non_fluents, 'init_non_fluent'):
            self.constants.update({nf: val for nf, val in self.model.non_fluents.init_non_fluent})
            logging.info(f'Initialized {len(self.constants)} constants from non-fluent definition')

        # try to initialize non-fluents from definition's default value
        for nf in domain.non_fluents.values():
            if nf.arity != 0:  # can't initialize parameterizable constants
                continue
            nf_name = (nf.name, None)
            if nf_name not in self.constants:  # non-fluent definition on file takes precedence
                self.constants[nf_name] = nf.default
                logging.info(f'Initialized constant "{nf_name}" with default value "{nf.default}"')
        logging.info(f'Total {len(self.constants)} constants initialized: {self.constants}')

        # create variables from state fluents
        logging.info('__________________________________________________')
        self.fluent_to_feature = {}
        for sf in domain.state_fluents.values():
            f_name = f'{sf.name}'
            f = self.world.defineState(agent.name, f_name, type(sf.default))
            self.world.setFeature(f, sf.default)
            self.fluent_to_feature[sf.name] = f
            logging.info(f'Created feature "{f}" from state fluent "{sf.name}"')
        logging.info(f'Total {len(self.fluent_to_feature)} features created')

        # create actions for agent
        logging.info('__________________________________________________')
        self.actions = {agent_name: {act.name: agent.addAction({'verb': act.name})}
                        for act in domain.action_fluents.values()}
        logging.info(f'Total {len(self.actions)} actions created from action fluent domain: {self.actions}')

        # create reward function
        logging.info('__________________________________________________')
        self._set_reward(self.model.domain.reward, agent)
        logging.info(f'Set agent "{agent.name}" reward to: {agent.getReward(agent.get_true_model())}')
        print('Done!')

    def _is_feature(self, sf: str) -> bool:
        # todo n-arity
        return sf in self.fluent_to_feature

    def _get_feature(self, sf: str) -> str:
        # todo n-arity
        return self.fluent_to_feature[sf]

    def _is_constant(self, nf: str) -> bool:
        return any(c[0] == nf for c in self.constants.keys())

    def _get_constant_value(self, nf: str) -> object:
        # todo n-arity
        return next(val for c, val in self.constants.items() if c[0] == nf)

    def _set_reward(self, expression: Expression, agent: Agent):
        return agent.setReward(makeTree(KeyedMatrix(
            {makeFuture(rewardKey(agent.name)): KeyedVector(self._get_expression_weights(expression))})), 1.)

    def _get_expression_weights(self, expression: Expression) -> Dict[str, float]:
        # process leaf node, try to get feature name or constant value
        e_type = expression.etype[0]
        if e_type == 'constant':
            try:
                return float(expression.args)
            except ValueError:
                logging.info(f'Could not convert value "{expression.args}" to float in RDDL expression "{expression}"')

        if e_type == 'pvar':
            name = expression.args[0]
            if self._is_feature(name):
                return {name: 1.}
            if self._is_constant(name):
                try:
                    value = self._get_constant_value(name)
                    return {CONSTANT: float(value)}
                except ValueError as e:
                    logging.info(f'Could not convert value "{value}" to float in RDDL expression "{expression}"')
            raise ValueError(f'Could not find feature or constant from RDDL expression "{expression}"')

        # process arithmetic node
        if e_type == 'arithmetic':
            def _get_const_val(s):
                return s if isinstance(s, float) else s[CONSTANT] if (len(s) == 1 and CONSTANT in s) else None

            def _update_weights(s):
                for k, v in s.items():
                    weights[k] = weights[k] + v if k in weights else v  # add weight if key already in dict

            lhs = self._get_expression_weights(expression.args[0])
            rhs = self._get_expression_weights(expression.args[1])
            lhs_const = _get_const_val(lhs)
            rhs_const = _get_const_val(rhs)
            all_consts = isinstance(lhs_const, float) and isinstance(rhs_const, float)

            weights = {}
            a_type = expression.etype[1]
            if a_type == '+':
                if all_consts:
                    weights.update({CONSTANT: lhs_const + rhs_const})  # reduce
                else:
                    _update_weights(lhs)  # if addition, just add everything both sides
                    _update_weights(rhs)
            elif a_type == '-':
                if all_consts:
                    weights.update({CONSTANT: lhs_const - rhs_const})  # reduce
                else:
                    _update_weights(lhs)  # if subtraction, first get left-hand side
                    _update_weights({k: -v for k, v in rhs.items()})  # then multiply right-hand side by -1
            elif a_type == '*':
                # if multiplication, only works if one or both sides are constants
                if all_consts:
                    weights.update({CONSTANT: lhs_const * rhs_const})  # reduce
                elif isinstance(lhs_const, float):
                    _update_weights({k: lhs_const * v for k, v in rhs.items()})  # multiply right-hand side by const
                elif isinstance(rhs_const, float):
                    _update_weights({k: rhs_const * v for k, v in lhs.items()})  # multiply left-hand side by const
                else:
                    raise ValueError(f'Non-PWL operation not supported: "{expression}"!')
            elif a_type == '/':
                # if division, only works if right or both sides are constants
                if all_consts:
                    weights.update({CONSTANT: lhs_const / rhs_const})  # reduce
                elif isinstance(rhs_const, float):
                    _update_weights({k: v / rhs_const for k, v in lhs.items()})  # divide left-hand side by const
                else:
                    raise ValueError(f'Non-PWL operation not supported: "{expression}"!')
            return weights

        # not yet implemented
        raise NotImplementedError(f'Cannot parse expression: "{expression}" of type "{e_type}"!')
