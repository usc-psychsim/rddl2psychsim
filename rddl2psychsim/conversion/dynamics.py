import logging
from typing import Dict
from pyrddl.expr import Expression
from psychsim.pwl import KeyedVector, rewardKey, makeTree, makeFuture, KeyedMatrix, KeyedTree, KeyedPlane, \
    CONSTANT, noChangeMatrix
from rddl2psychsim.conversion.expression import _ExpressionConverter, _is_linear_function, _combine_linear_functions, \
    _negate_linear_function, _get_const_val

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class _DynamicsConverter(_ExpressionConverter):

    def __init__(self):
        super().__init__()

    def _convert_dynamics(self):
        # create dynamics from conditional probability functions (CPFs)
        logging.info('__________________________________________________')
        for cpf in self.model.domain.cpfs[1]:
            f_type = cpf.pvar[0]
            if f_type == 'pvar_expr':
                name = cpf.pvar[1][0].replace('\'', '')
                if cpf.pvar[1][1] is None:
                    params = []
                    param_vals = [(None,)]
                else:
                    # gets all combinations for features
                    params = [p for p in cpf.pvar[1][1]]
                    param_types = self._get_param_types(name)
                    param_vals = self._get_all_param_combs(param_types)

                # sets dynamics for each feature
                for p_vals in param_vals:
                    f_name = (name, *p_vals)
                    if not self._is_feature(f_name):
                        raise ValueError(f'Could not find feature for fluent "{f_name}" in CPF "{cpf}"!')
                    f = self._get_feature(f_name)
                    tree = self._create_dynamics_tree(f, cpf.expr, dict(zip(params, p_vals)))
                    self.world.setDynamics(f, True, tree)
                    logging.info(f'Set dynamics for feature "{f}" to:\n{tree}')
            else:
                raise NotImplementedError(f'Cannot convert CPF "{cpf}" of type "{f_type}"!')

    def _convert_reward_function(self):
        # create reward function # TODO assume homogeneous agents
        logging.info('__________________________________________________')
        for agent in self.world.agents.values():
            tree = self._create_dynamics_tree(rewardKey(agent.name), self.model.domain.reward)
            agent.setReward(tree, 1.)
            logging.info(f'Set agent "{agent.name}" reward to:\n{tree}')

    def _create_dynamics_tree(self, key: str, expression: Expression, param_map: Dict[str, str] = None) -> KeyedTree:
        return makeTree(self._get_dynamics_tree(key, self._convert_expression(expression, param_map)))

    def _get_dynamics_tree(self, key: str, expr: Dict) -> KeyedMatrix or Dict:

        # just get the truth value of logical expressions
        op = next(iter(expr.keys()))
        if len(expr) == 1 and op in \
                {'linear_and', 'logic_and', 'linear_or', 'logic_or', 'not', 'equiv', 'imply',
                 'eq', 'neq', 'gt', 'lt', 'geq', 'leq',
                 'action'}:
            if _get_const_val(expr[op]) is not None:
                return self._get_dynamics_tree(key, expr[op])  # no need for tree if it's a constant value
            return self._get_dynamics_tree(key, self._get_pwl_tree(expr, {CONSTANT: True}, {CONSTANT: False}))

        if 'if' in expr and len(expr) == 3:
            # check if no comparison provided, expression's truth value has to be resolved
            if len(expr['if']) == 1:
                return self._get_dynamics_tree(key, self._get_pwl_tree(expr['if'], expr[True], expr[False]))

            # build if-then-else tree
            weights, threshold, comp = expr['if']
            return {'if': KeyedPlane(KeyedVector(weights), threshold, comp),
                    True: self._get_dynamics_tree(key, expr[True]),
                    False: self._get_dynamics_tree(key, expr[False])}

        if 'switch' in expr and len(expr) == 1:
            cond, case_values, case_branches = expr['switch']

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
                c = _combine_linear_functions(cond, _negate_linear_function(val))  # gets difference between planes
                tree['if'] = KeyedPlane(KeyedVector(c), 0, 0)  # tests equality
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

        if 'distribution' in expr and len(expr) == 1:
            # create stochastic effect
            return {'distribution': [(KeyedMatrix({makeFuture(key): KeyedVector(v)}), p)
                                     for v, p in expr['distribution']]}

        # if all key-value pairs, assume direct linear combination of all features
        if _is_linear_function(expr) or self._is_enum_expr(expr):
            return KeyedMatrix({makeFuture(key): KeyedVector({CONSTANT: 0} if len(expr) == 0 else expr)})

        raise NotImplementedError(f'Could not parse RDDL expression, got invalid tree: "{expr}"!')
