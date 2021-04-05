import logging
from typing import Dict
from pyrddl.expr import Expression
from psychsim.agent import Agent
from psychsim.pwl import KeyedVector, rewardKey, makeTree, makeFuture, KeyedMatrix, KeyedTree, KeyedPlane, \
    setToConstantMatrix
from rddl2psychsim.conversion.expression import _ExpressionConverter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class _DynamicsConverter(_ExpressionConverter):

    def __init__(self):
        super().__init__()

    def _convert_dynamics(self, agent):
        # create dynamics from conditional probability functions (CPFs)
        logging.info('__________________________________________________')
        for cpf in self.model.domain.cpfs[1]:
            f_type = cpf.pvar[0]
            if f_type == 'pvar_expr':
                # todo n-arity
                name = cpf.pvar[1][0].replace('\'', '')
                if not self._is_feature(name):
                    raise ValueError(f'Could not find feature for fluent "{name}" in CPF "{cpf}"!')
                f = self._get_feature(name)
                tree = self._create_dynamics_tree(f, cpf.expr, agent)
                self.world.setDynamics(f, True, tree)
                logging.info(f'Set dynamics for feature "{f}" to:\n{tree}')
            else:
                raise NotImplementedError(f'Cannot convert CPF "{cpf}" of type "{f_type}"!')

    def _convert_reward_function(self, agent):
        # create reward function
        logging.info('__________________________________________________')
        tree = self._create_dynamics_tree(rewardKey(agent.name), self.model.domain.reward, agent)
        agent.setReward(tree, 1.)
        logging.info(f'Set agent "{agent.name}" reward to:\n{tree}')

    def _create_dynamics_tree(self, key, expression: Expression, agent: Agent) -> KeyedTree:
        return makeTree(self._get_dynamics_tree(key, self._get_expression_dict(expression, agent)))

    def _get_dynamics_tree(self, key, tree_dict) -> KeyedMatrix or Dict:
        if all(isinstance(k, str) for k in tree_dict.keys()) and all(isinstance(v, float) for v in tree_dict.values()):
            # if all key-value pairs, create a keyed vector
            return KeyedMatrix({makeFuture(key): KeyedVector(tree_dict)})

        if 'if' in tree_dict:
            # build if-then-else tree
            weights, threshold, comp = tree_dict['if']
            return {'if': KeyedPlane(KeyedVector(weights), threshold, comp),
                    True: self._get_dynamics_tree(key, tree_dict[True]),
                    False: self._get_dynamics_tree(key, tree_dict[False])}

        if 'distribution' in tree_dict:
            # create stochastic effect
            return {'distribution': [(setToConstantMatrix(key, v), p) for v, p in tree_dict['distribution']]}

        raise NotImplementedError(f'Could not parse RDDL expression, got invalid tree: "{tree_dict}"!')
