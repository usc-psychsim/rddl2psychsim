import logging
from typing import Dict, Tuple
from pyrddl.expr import Expression
from psychsim.action import ActionSet
from psychsim.pwl import KeyedVector, rewardKey, makeTree, makeFuture, KeyedMatrix, KeyedPlane, CONSTANT, \
    noChangeMatrix, actionKey
from rddl2psychsim.conversion.expression import _ExpressionConverter, _is_linear_function, _combine_linear_functions, \
    _negate_linear_function, _get_const_val

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class _DynamicsConverter(_ExpressionConverter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                    self._create_dynamics(f, cpf.expr, dict(zip(params, p_vals)))
            else:
                raise NotImplementedError(f'Cannot convert CPF "{cpf}" of type "{f_type}"!')

    def _convert_reward_function(self):
        # create reward function # TODO assume homogeneous agents
        logging.info('__________________________________________________')
        for agent in self.world.agents.values():
            expr = self._convert_expression(self.model.domain.reward)
            tree = makeTree(self._get_dynamics_tree(rewardKey(agent.name), expr))
            agent.setReward(tree, 1.)
            logging.info(f'Set agent "{agent.name}" reward to:\n{tree}')

    def _create_dynamics(self, key: str, expression: Expression, param_map: Dict[str, str] = None) -> None:
        # tries to get actions responsible for feature update from expression
        expr = self._convert_expression(expression, param_map)
        action_dynamics = self._extract_action_dynamics(expr)

        # for each sub-expression, set dynamics for feature as provided by an action (or the world itself)
        for action, dyn_expr in action_dynamics.items():
            tree = makeTree(self._get_dynamics_tree(key, dyn_expr))
            self.world.setDynamics(key, action, tree)
            logging.info(f'Set dynamics for feature "{key}" associated with "{action}" to:\n{tree}')

    def _extract_action_dynamics(self, expression: Dict) -> Dict[ActionSet or bool, Dict]:
        # check if we can pull actions from the sub-expressions
        if 'if' in expression and 'action' in expression['if']:
            # assigns dynamics to action and processes rest of expression
            _, action, _ = expression['if']['action']
            dynamics = {action: expression[True]}
            dynamics.update(self._extract_action_dynamics(expression[False]))
            return dynamics
        else:
            return {True: expression}  # otherwise assign dynamics to world

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

    def _get_pwl_tree(self, comp: Dict, true_branch, false_branch) -> Dict:
        if 'linear_and' in comp and len(comp) == 1:
            comp = comp['linear_and']  # AND of features (sum > w_sum - 0.5), see psychsim.world.World.float2value
            return {'if': (comp, sum([v for v in comp.values() if v > 0]) - 0.5, 1),
                    True: true_branch,
                    False: false_branch}

        if 'logic_and' in comp and len(comp) == 1:
            lhs, rhs = comp['logic_and']  # composes nested AND tree
            return {'if': lhs,
                    True: self._get_pwl_tree(rhs, true_branch, false_branch),
                    False: false_branch}

        if 'linear_or' in comp and len(comp) == 1:
            comp = comp['linear_or']  # OR of features (A | B) <=> ~(~A ^ ~B)
            return self._get_pwl_tree({'linear_and': _negate_linear_function(comp)},
                                      false_branch,  # switch branches
                                      true_branch)

        if 'logic_or' in comp and len(comp) == 1:
            lhs, rhs = comp['logic_or']  # composes nested OR tree
            return {'if': lhs,
                    True: true_branch,
                    False: self._get_pwl_tree(rhs, true_branch, false_branch)}

        if 'not' in comp and len(comp) == 1:
            # if NOT, just flip branches
            return self._get_pwl_tree(comp['not'], false_branch, true_branch)

        if 'equiv' in comp:
            # if logical EQUIVALENCE, true iff both sides are true or both are false
            lhs, rhs = comp['equiv']
            return self._get_pwl_tree(lhs,
                                      self._get_pwl_tree(rhs, true_branch, false_branch),
                                      self._get_pwl_tree(rhs, false_branch, true_branch))

        if 'imply' in comp:
            # if IMPLICATION, false only if left is true and right is false
            lhs, rhs = comp['imply']
            return self._get_pwl_tree(lhs,
                                      self._get_pwl_tree(rhs, true_branch, false_branch),
                                      true_branch)

        if 'eq' in comp:
            lhs, rhs = comp['eq']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, 0),  # takes equality of pwl comb in vectors (difference==0)
                    True: true_branch,
                    False: false_branch}

        if 'neq' in comp:
            lhs, rhs = comp['neq']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, 0),  # takes equality of pwl comb in vectors (difference==0)
                    True: false_branch,  # then switch branches
                    False: true_branch}

        if 'gt' in comp:
            lhs, rhs = comp['gt']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, 1),  # takes diff of pwl comb in vectors (difference>0)
                    True: true_branch,
                    False: false_branch}

        if 'lt' in comp:
            lhs, rhs = comp['lt']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, -1),  # takes diff of pwl comb in vectors (difference<0)
                    True: true_branch,
                    False: false_branch}

        if 'geq' in comp:
            lhs, rhs = comp['geq']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, -1),  # takes diff of pwl comb in vectors (difference<0)
                    True: false_branch,  # then switch branches
                    False: true_branch}

        if 'leq' in comp:
            lhs, rhs = comp['leq']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, 1),  # takes diff of pwl comb in vectors (difference>0)
                    True: false_branch,  # then switch branches
                    False: true_branch}

        if 'action' in comp and len(comp['action']) == 3:
            agent, action, future = comp['action']
            key = makeFuture(actionKey(agent.name)) if future else actionKey(agent.name)  # check future vs prev action
            return {'if': ({key: 1}, action, 0),  # conditional on specific agent's action
                    True: true_branch,
                    False: false_branch}

        if _is_linear_function(comp):
            # default: assumes linear combination of all features in vector has to be > 0.5,
            # which is truth value in PsychSim (see psychsim.world.World.float2value)
            return {'if': (comp, 0.5, 1),
                    True: true_branch,
                    False: false_branch}

        raise ValueError(f'Could not parse RDDL expression, unknown PWL tree comparison "{comp}"!')

    def _get_relational_plane_thresh(self, lhs: Dict, rhs: Dict) -> Tuple[Dict, str or int]:
        if _is_linear_function(lhs) and _is_linear_function(rhs):
            op = _combine_linear_functions(lhs, _negate_linear_function(rhs))
            return op, 0  # if both sides are planes, returns difference (threshold 0)

        if _is_linear_function(lhs) and self._is_enum_expr(rhs):
            return lhs, rhs[CONSTANT]  # if comparison with enum, return enum value as threshold

        if self._is_enum_expr(lhs) and _is_linear_function(rhs):
            return rhs, lhs[CONSTANT]  # if comparison with enum, return enum value as threshold

        raise ValueError(f'Cannot parse expression, non-PWL relational operation between {lhs} and {rhs}!')
