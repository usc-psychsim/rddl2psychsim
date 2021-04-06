import logging
from typing import Dict
from pyrddl.expr import Expression
from psychsim.agent import Agent
from psychsim.pwl import KeyedVector, rewardKey, makeTree, makeFuture, KeyedMatrix, KeyedTree, KeyedPlane, \
    setToConstantMatrix, CONSTANT
from rddl2psychsim.conversion.expression import _ExpressionConverter, is_pwl_op, nested_if

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

    def _create_dynamics_tree(self, key: str, expression: Expression, agent: Agent) -> KeyedTree:
        return makeTree(self._get_dynamics_tree(key, self._get_expression_dict(expression, agent)))

    def _get_dynamics_tree(self, key: str, tree_dict: Dict) -> KeyedMatrix or Dict:

        # just get the truth value of logical expressions
        if len(tree_dict) == 1 and next(iter(tree_dict.keys())) in \
                {'pwl_and', 'logic_and', 'pwl_or', 'logic_or', 'not', 'equiv', 'imply',
                 'eq', 'neq', 'gt', 'lt', 'geq', 'leq'}:
            return self._get_dynamics_tree(key, nested_if(tree_dict, {CONSTANT: True}, {CONSTANT: False}))

        if 'if' in tree_dict and len(tree_dict) == 3:
            # build if-then-else tree
            weights, threshold, comp = tree_dict['if']
            return {'if': KeyedPlane(KeyedVector(weights), threshold, comp),
                    True: self._get_dynamics_tree(key, tree_dict[True]),
                    False: self._get_dynamics_tree(key, tree_dict[False])}

        if 'distribution' in tree_dict and len(tree_dict) == 1:
            # create stochastic effect
            return {'distribution': [(setToConstantMatrix(key, v), p) for v, p in tree_dict['distribution']]}

        # if all key-value pairs, assume linear combination of all features
        if is_pwl_op(tree_dict):
            return KeyedMatrix({makeFuture(key): KeyedVector(tree_dict)})

        raise NotImplementedError(f'Could not parse RDDL expression, got invalid tree: "{tree_dict}"!')
