import logging
from typing import Dict
from pyrddl.expr import Expression
from psychsim.agent import Agent
from psychsim.pwl import KeyedVector, rewardKey, makeTree, makeFuture, KeyedMatrix, KeyedTree, KeyedPlane, \
    setToConstantMatrix, CONSTANT, noChangeMatrix
from rddl2psychsim.conversion.expression import _ExpressionConverter, is_pwl_op, nested_if, update_weights, \
    negate_weights

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
                print(f'Set dynamics for feature "{f}" to:\n{tree}')
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

    def _get_dynamics_tree(self, key: str, expr_dict: Dict) -> KeyedMatrix or Dict:

        # just get the truth value of logical expressions
        if len(expr_dict) == 1 and next(iter(expr_dict.keys())) in \
                {'pwl_and', 'logic_and', 'pwl_or', 'logic_or', 'not', 'equiv', 'imply',
                 'eq', 'neq', 'gt', 'lt', 'geq', 'leq'}:
            return self._get_dynamics_tree(key, nested_if(expr_dict, {CONSTANT: True}, {CONSTANT: False}))

        if 'if' in expr_dict and len(expr_dict) == 3:
            # build if-then-else tree
            weights, threshold, comp = expr_dict['if']
            return {'if': KeyedPlane(KeyedVector(weights), threshold, comp),
                    True: self._get_dynamics_tree(key, expr_dict[True]),
                    False: self._get_dynamics_tree(key, expr_dict[False])}

        if 'switch' in expr_dict and len(expr_dict) == 1:
            cond, case_values, case_branches = expr_dict['switch']

            # separates conditions and branches for constants and other conditional values
            const_values, const_branches = [], []
            if_values, if_branches = [], []
            def_branch = noChangeMatrix(key)  # default is don't change key
            for i, val in enumerate(case_values):
                const = val[CONSTANT] if len(val) == 1 and CONSTANT in val else None
                branch = case_branches[i]
                if val == 'default':
                    def_branch = self._get_dynamics_tree(key, branch)
                elif const is None:
                    if_values.append(val)
                    if_branches.append(branch)
                else:
                    const_values.append(const)
                    const_branches.append(branch)

            # first get nested if tree (if needed)
            root = tree = {}
            for i, val in enumerate(if_values):
                if False in tree:
                    tree = tree[False]  # nest if
                c = dict(cond)
                update_weights(c, negate_weights(val))  # gets difference between planes
                tree['if'] = KeyedPlane(KeyedVector(c), 0, 0)  # tests PWL equality
                tree[True] = self._get_dynamics_tree(key, if_branches[i])
                tree[False] = {}

            # if no cases for constant conditionals, return if tree
            if len(const_branches) == 0:
                if False in tree:
                    tree[False] = def_branch
                else:
                    root = def_branch
                return root

            if False in tree:
                tree = tree[False]  # nest if

            # then get switch-case tree (if needed)
            tree['if'] = KeyedPlane(KeyedVector(cond), const_values, 0)  # compares equality against const values
            for i, branch in enumerate(const_branches):
                tree[i] = self._get_dynamics_tree(key, branch)
            tree[None] = def_branch
            return root

        if 'distribution' in expr_dict and len(expr_dict) == 1:
            # create stochastic effect
            return {'distribution': [(setToConstantMatrix(key, v), p) for v, p in expr_dict['distribution']]}

        # if all key-value pairs, assume direct linear combination of all features
        if is_pwl_op(expr_dict):
            return KeyedMatrix({makeFuture(key): KeyedVector(expr_dict)})

        raise NotImplementedError(f'Could not parse RDDL expression, got invalid tree: "{expr_dict}"!')
