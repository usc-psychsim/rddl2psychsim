import logging
import numpy as np
from functools import reduce
from typing import Dict, Tuple, Set
from pyrddl.expr import Expression
from psychsim.action import ActionSet
from psychsim.pwl import KeyedVector, rewardKey, makeTree, makeFuture, KeyedMatrix, KeyedPlane, CONSTANT, \
    noChangeMatrix, actionKey
from rddl2psychsim.conversion.expression import _ExpressionConverter, _is_linear_function, _combine_linear_functions, \
    _negate_linear_function, _get_const_val

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

# used in float comparison to be equivalent to <=, >= since PsychSim does not support these
EPS = np.finfo(np.float64).eps


class _DynamicsConverter(_ExpressionConverter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_dependencies(self, f_name: str) -> Set[str]:
        if f_name not in self._fluent_levels:
            return set()

        # adds fluents of level below as dependencies
        level = self._fluent_levels[f_name]
        return set([f for f, l in self._fluent_levels.items() if f != f_name and l < level])

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
                    dependencies = self._get_dependencies(name)
                    self._create_dynamics(self._get_feature(f_name), cpf.expr, dict(zip(params, p_vals)), dependencies)
            else:
                raise NotImplementedError(f'Cannot convert CPF "{cpf}" of type "{f_type}"!')

    def _convert_reward_function(self):
        # create reward function # TODO assume homogeneous agents
        logging.info('__________________________________________________')
        for agent in self.world.agents.values():
            expr = self._convert_expression(self.model.domain.reward, dependencies=set())
            tree = makeTree(self._get_dynamics_tree(rewardKey(agent.name), expr))
            agent.setReward(tree, 1.)
            logging.info(f'Set agent "{agent.name}" reward to:\n{tree}')

    def _create_dynamics(self, key: str, expression: Expression,
                         param_map: Dict[str, str] or None = None,
                         dependencies: Set[str] = None) -> None:
        # tries to get actions responsible for feature update from expression
        expr = self._convert_expression(expression, param_map, dependencies)
        action_dynamics = self._extract_action_dynamics(expr)

        # for each sub-expression, set dynamics for feature as provided by an action (or the world itself)
        for action, dyn_expr in action_dynamics.items():
            tree = makeTree(self._get_dynamics_tree(key, dyn_expr))
            self.world.setDynamics(key, action, tree)
            logging.info(f'Set dynamics for feature "{key}" associated with "{action}" to:\n{tree}')

    def _extract_action_dynamics(self, expression: Dict) -> Dict[ActionSet or bool, Dict]:

        def _get_act_disjunction(expr):
            if 'action' in expr:
                _, action, _ = expr['action']  # add to list of actions
                actions.add(action)
                return True
            if 'logic_or' in expr:
                return all(_get_act_disjunction(sub_expr) for sub_expr in expr['logic_or'])
            return False  # top expression is not a disjunction of actions

        # check if we can pull actions from the sub-expressions in the form "if (action [or action...]) then dyn_expr"
        actions = set()
        if 'if' in expression and _get_act_disjunction(expression['if']):
            # assigns dynamics to actions and processes rest of expression
            dynamics = {action: expression[True] for action in actions}
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
            if isinstance(expr['if'], dict) and len(expr['if']) == 1:
                return self._get_dynamics_tree(key, self._get_pwl_tree(expr['if'], expr[True], expr[False]))

            assert isinstance(expr['if'], KeyedPlane), f'Could not parse RDDL expression, got invalid tree: "{expr}"!'

            # build if-then-else tree
            return {'if': expr['if'],
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
        if _is_linear_function(expr) or self._is_constant_expr(expr):
            return KeyedMatrix({makeFuture(key): KeyedVector({CONSTANT: 0} if len(expr) == 0 else expr)})

        raise NotImplementedError(f'Could not parse RDDL expression, got invalid tree: "{expr}"!')

    def _get_pwl_tree(self, expr: Dict, true_branch, false_branch) -> Dict:
        # first tries to get plane directly from expression
        plane = self._get_plane(expr)
        if plane is not None:
            # if we have a valid plane, then simply return tree
            return {'if': plane,
                    True: true_branch,
                    False: false_branch}

        # otherwise we have to build (possibly nested) PWL trees
        if 'logic_and' in expr and len(expr) == 1:
            sub_exprs = expr['logic_and']
            assert len(sub_exprs) > 1, f'Could not parse RDDL expression, AND needs at least two arguments "{expr}"!'
            lhs = sub_exprs[0]
            rhs = {'logic_and': sub_exprs[1:]} if len(sub_exprs) > 2 else sub_exprs[1]
            return self._get_pwl_tree(lhs,
                                      self._get_pwl_tree(rhs, true_branch, false_branch),
                                      false_branch)

        if 'logic_or' in expr and len(expr) == 1:
            sub_exprs = expr['logic_or']
            assert len(sub_exprs) > 1, f'Could not parse RDDL expression, OR needs at least two arguments "{expr}"!'
            lhs = sub_exprs[0]
            rhs = {'logic_or': sub_exprs[1:]} if len(sub_exprs) > 2 else sub_exprs[1]
            return self._get_pwl_tree(lhs,
                                      true_branch,
                                      self._get_pwl_tree(rhs, true_branch, false_branch))

        if 'not' in expr and len(expr) == 1:
            # if NOT, just flip branches
            return self._get_pwl_tree(expr['not'], false_branch, true_branch)

        if 'equiv' in expr:
            # if logical EQUIVALENCE, true iff both sides are true or both are false
            lhs, rhs = expr['equiv']
            return self._get_pwl_tree(lhs,
                                      self._get_pwl_tree(rhs, true_branch, false_branch),
                                      self._get_pwl_tree(rhs, false_branch, true_branch))

        if 'imply' in expr:
            # if IMPLICATION, false only if left is true and right is false
            lhs, rhs = expr['imply']
            return self._get_pwl_tree(lhs,
                                      self._get_pwl_tree(rhs, true_branch, false_branch),
                                      true_branch)

        raise ValueError(f'Could not parse RDDL expression, unknown PWL tree comparison "{expr}"!')

    def _get_plane(self, expr: Dict, negate: bool = False) -> KeyedPlane or None:
        # KeyedPlane(KeyedVector(weights), threshold, comparison)

        if 'not' in expr and len(expr) == 1:
            # if NOT, get opposite operation
            return self._get_plane(expr['not'], negate=not negate)

        if _is_linear_function(expr):
            # assumes linear combination of all features in vector has to be > 0.5,
            # which is truth value in PsychSim (see psychsim.world.World.float2value)
            return KeyedPlane(KeyedVector(expr), 0.5 + EPS, -1) if negate else \
                KeyedPlane(KeyedVector(expr), 0.5, 1)

        if 'action' in expr and len(expr['action']) == 3:
            # conditional on specific agent's action (agent's action == the action)
            agent, action, future = expr['action']
            key = makeFuture(actionKey(agent.name)) if future else actionKey(agent.name)  # check future vs prev action
            if negate:
                return KeyedPlane(KeyedVector({key: 1}), action, 1) | KeyedPlane(KeyedVector({key: 1}), action, -1)
            else:
                return KeyedPlane(KeyedVector({key: 1}), action, 0)

        if 'linear_and' in expr and len(expr) == 1:
            # AND of features (sum > w_sum - 0.5), see psychsim.world.World.float2value
            # for negation, ~(A ^ B ^ ...) <=> ~A | ~B | ...
            expr = expr['linear_and']
            if negate:
                return self._get_plane({'linear_or': _negate_linear_function(expr)})
            else:
                return KeyedPlane(KeyedVector(expr), sum([v for v in expr.values() if v > 0]) - 0.5, 1)

        if 'linear_or' in expr and len(expr) == 1:
            # OR of features (A | B | ...) <=> ~(~A ^ ~B ^ ...)
            # for negation, ~(A | B | ...) <=> ~A ^ ~B ^ ...
            expr = expr['linear_or']
            if negate:
                return self._get_plane({'linear_and': _negate_linear_function(expr)})
            else:
                expr = _negate_linear_function(expr)
                return KeyedPlane(KeyedVector(expr), sum([v for v in expr.values() if v > 0]) - 0.5 + EPS, -1)

        if 'logic_and' in expr and len(expr) == 1:
            # get conjunction between sub-expressions (all planes must be valid)
            # for negation, ~(A ^ B ^ ...) <=> ~A | ~B |...
            sub_exprs = expr['logic_and']
            if negate:
                planes = [self._get_plane({'not': sub_expr}) for sub_expr in sub_exprs]
                invalid = None in planes or any(len(plane.planes) > 1 and plane.isConjunction for plane in planes)
                return None if invalid else reduce(lambda p1, p2: p1 | p2, planes)  # disjunction
            else:
                planes = [self._get_plane(sub_expr) for sub_expr in sub_exprs]
                invalid = None in planes or any(len(plane.planes) > 1 and not plane.isConjunction for plane in planes)
                return None if invalid else reduce(lambda p1, p2: p1 & p2, planes)  # conjunction

        if 'logic_or' in expr and len(expr) == 1:
            # get disjunction between sub-expressions (all planes must be valid)
            # for negation, ~(A | B | ...) <=> ~A ^ ~B ^ ...
            sub_exprs = expr['logic_or']
            if negate:
                planes = [self._get_plane({'not': sub_expr}) for sub_expr in sub_exprs]
                invalid = None in planes or any(len(plane.planes) > 1 and not plane.isConjunction for plane in planes)
                return None if invalid else reduce(lambda p1, p2: p1 & p2, planes)  # conjunction
            else:
                planes = [self._get_plane(sub_expr) for sub_expr in sub_exprs]
                invalid = None in planes or any(len(plane.planes) > 1 and plane.isConjunction for plane in planes)
                return None if invalid else reduce(lambda p1, p2: p1 | p2, planes)  # disjunction

        # test binary operators
        op = next(iter(expr.keys()))
        if len(expr) == 1 and op in {'eq', 'neq', 'gt', 'lt', 'geq', 'leq', 'imply'}:
            lhs, rhs = expr[op]
            if (not negate and 'eq' in expr) or (negate and 'neq' in expr):
                # takes equality of pwl comb in vectors (difference==0 or expr==threshold)
                weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                return KeyedPlane(KeyedVector(weights), thresh, 0)

            if (not negate and 'neq' in expr) or (negate and 'eq' in expr):
                # takes equality of pwl comb in vectors (difference!=0 or expr!=threshold)
                weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                return KeyedPlane(KeyedVector(weights), thresh, 1) | KeyedPlane(KeyedVector(weights), thresh, -1)

            if (not negate and 'gt' in expr) or (negate and 'leq' in expr):
                # takes diff of pwl comb in vectors (difference>0 or expr>threshold)
                weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                return KeyedPlane(KeyedVector(weights), thresh, 1)

            if (not negate and 'lt' in expr) or (negate and 'geq' in expr):
                # takes diff of pwl comb in vectors (difference<0 or expr<threshold)
                weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                return KeyedPlane(KeyedVector(weights), thresh, -1)

            if (not negate and 'geq' in expr) or (negate and 'lt' in expr):
                # takes diff of pwl comb in vectors (difference>=0 or expr>=threshold)
                weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                if isinstance(thresh, str):
                    return KeyedPlane(KeyedVector(weights), thresh, 1) | KeyedPlane(KeyedVector(weights), thresh, 0)
                else:
                    return KeyedPlane(KeyedVector(weights), thresh - EPS, 1)

            if (not negate and 'leq' in expr) or (negate and 'gt' in expr):
                # takes diff of pwl comb in vectors (difference<=0 or expr<=threshold)
                weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                if isinstance(thresh, str):
                    KeyedPlane(KeyedVector(weights), thresh, -1) | KeyedPlane(KeyedVector(weights), thresh, 0)
                else:
                    return KeyedPlane(KeyedVector(weights), thresh + EPS, -1)

            if 'imply' in expr:
                # if IMPLICATION, false only if left is true and right is false, ie
                # true if left is false or right is true
                if not (_is_linear_function(lhs) and _is_linear_function(rhs)):
                    return None  # both operands have to be linear combinations
                if negate:
                    return KeyedPlane(KeyedVector(lhs), 0.5, 1) & KeyedPlane(KeyedVector(rhs), 0.5 + EPS, -1)
                else:
                    return KeyedPlane(KeyedVector(lhs), 0.5 + EPS, -1) | KeyedPlane(KeyedVector(rhs), 0.5, 1)

        return None

    def _get_relational_plane_thresh(self, lhs: Dict, rhs: Dict) -> Tuple[Dict, str or int]:
        if _is_linear_function(lhs) and _is_linear_function(rhs):
            op = _combine_linear_functions(lhs, _negate_linear_function(rhs))
            return op, 0  # if both sides are planes, returns difference (threshold 0)

        if _is_linear_function(lhs) and self._is_constant_expr(rhs):
            return lhs, rhs[CONSTANT]  # if comparison with enum, return enum value as threshold

        if self._is_constant_expr(lhs) and _is_linear_function(rhs):
            return rhs, lhs[CONSTANT]  # if comparison with enum, return enum value as threshold

        raise ValueError(f'Cannot parse expression, non-PWL relational operation between {lhs} and {rhs}!')
