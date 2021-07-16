import logging
import numpy as np
from functools import reduce
from typing import Dict, Tuple, Set, Callable, List
from pyrddl.expr import Expression
from psychsim.action import ActionSet
from psychsim.pwl import KeyedVector, rewardKey, makeTree, makeFuture, KeyedMatrix, KeyedPlane, CONSTANT, \
    actionKey, KeyedTree
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
            tree = self._get_dynamics_tree(rewardKey(agent.name), expr)
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
            tree = self._get_dynamics_tree(key, dyn_expr)
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

    def _get_dynamics_tree(self, key: str, expr: Dict) -> KeyedTree:

        def _leaf_func(leaf_expr: Dict) -> KeyedMatrix:
            if _is_linear_function(leaf_expr) or self._is_constant_expr(leaf_expr):
                return KeyedMatrix({makeFuture(key): KeyedVector({CONSTANT: 0} if len(leaf_expr) == 0 else leaf_expr)})
            raise NotImplementedError(f'Could not parse RDDL expression, got invalid subtree: "{leaf_expr}"!')

        # return a dynamics decision tree from the expression
        return self._get_decision_tree(expr, _leaf_func)

    def _get_decision_tree(self, expr: Dict, leaf_func: Callable[[Dict], Dict or KeyedMatrix]) -> KeyedTree:

        # check if root operation is a logical expression
        op = next(iter(expr.keys()))
        if len(expr) == 1 and op in \
                {'linear_and', 'logic_and', 'linear_or', 'logic_or', 'not', 'equiv', 'imply',
                 'eq', 'neq', 'gt', 'lt', 'geq', 'leq',
                 'action'}:
            if _get_const_val(expr[op]) is not None:
                # no need for tree if it's a constant value
                return self._get_decision_tree(expr[op], leaf_func)

            # otherwise return a tree that gets the truth value of the expression
            return self._get_decision_tree(self._get_if_tree(
                expr, {True: {CONSTANT: True}, False: {CONSTANT: False}}), leaf_func)

        if 'if' in expr and len(expr) > 1:
            # check if no plane (branch) provided, then expression's truth value has to be resolved
            if isinstance(expr['if'], dict) and len(expr['if']) == 1:
                return self._get_decision_tree(self._get_if_tree(
                    expr['if'], {child: expr[child] for child in expr if child != 'if'}), leaf_func)

            assert isinstance(expr['if'], KeyedPlane), f'Could not parse RDDL expression, got invalid branch: "{expr}"!'

            # otherwise just create a PsychSim decision tree
            tree = {child: self._get_decision_tree(expr[child], leaf_func) for child in expr if child != 'if'}
            tree['if'] = expr['if']
            return makeTree(tree)

        if 'switch' in expr and len(expr) == 1:
            return self._get_decision_tree(self._get_switch_tree(*expr['switch']), leaf_func)

        if 'distribution' in expr and len(expr) == 1:
            # create stochastic tree
            return makeTree({'distribution': [(leaf_func(v), p) for v, p in expr['distribution']]})

        # just return expression's value
        return makeTree(leaf_func(expr))

    def _get_if_tree(self, branch: Dict, children: Dict) -> Dict:

        def _assert_boolean_tree():
            assert len(children) == 2 and True in children and False in children, \
                f'Could not parse RDDL expression, boolean tree needs True and False children: {children}'

        # first check if condition is constant
        const_val = _get_const_val(branch, bool)
        if const_val is not None:
            _assert_boolean_tree()
            return children[True] if const_val else children[False]

        # then tries to get branching plane directly from condition
        plane = self._get_plane(branch)
        if plane is not None:
            # if we have a valid plane, then simply return if tree
            return {'if': plane, **children}

        # otherwise we have to build (possibly nested) PWL trees
        if 'logic_and' in branch and len(branch) == 1:
            sub_exprs = branch['logic_and']
            assert len(sub_exprs) > 1, f'Could not parse RDDL expression, AND needs at least two arguments "{branch}"!'
            lhs = sub_exprs[0]
            rhs = {'logic_and': sub_exprs[1:]} if len(sub_exprs) > 2 else sub_exprs[1]
            _assert_boolean_tree()
            return self._get_if_tree(lhs,
                                     {True: self._get_if_tree(rhs, children),
                                      False: children[False]})

        if 'logic_or' in branch and len(branch) == 1:
            sub_exprs = branch['logic_or']
            assert len(sub_exprs) > 1, f'Could not parse RDDL expression, OR needs at least two arguments "{branch}"!'
            lhs = sub_exprs[0]
            rhs = {'logic_or': sub_exprs[1:]} if len(sub_exprs) > 2 else sub_exprs[1]
            _assert_boolean_tree()
            return self._get_if_tree(lhs,
                                     {True: children[True],
                                      False: self._get_if_tree(rhs, children)})

        if 'not' in branch and len(branch) == 1:
            # if NOT, just swap branches
            _assert_boolean_tree()
            return self._get_if_tree(branch['not'], {True: children[False], False: children[True]})

        if 'equiv' in branch:
            # if logical EQUIVALENCE, true iff both sides are true or both are false
            lhs, rhs = branch['equiv']
            _assert_boolean_tree()
            return self._get_if_tree(lhs,
                                     {True: self._get_if_tree(rhs, children),
                                      False: self._get_if_tree(rhs, {True: children[False], False: children[True]})})

        if 'imply' in branch:
            # if IMPLICATION, false only if left is true and right is false
            lhs, rhs = branch['imply']
            _assert_boolean_tree()
            return self._get_if_tree(lhs,
                                     {True: self._get_if_tree(rhs, children),
                                      False: children[True]})

        if 'switch' in branch:
            # if SWITCH, then propagate "if" children at each case child
            switch_branch, case_values, case_children = branch['switch']
            case_if_children = []
            for case_child in case_children:  # case child used as branch for the if subtree
                case_if_children.append({'if': case_child, **children})
            return {'switch': (switch_branch, case_values, case_if_children)}

        raise ValueError(f'Could not parse RDDL expression, unknown PWL tree comparison "{branch}"!')

    def _get_switch_tree(self, branch: Dict, case_values: List[Dict], case_children: List[Dict]) -> Dict:

        # separates conditions for constants and other conditional values
        const_values, const_children = [], []
        if_values, if_children = [], []
        def_child = None
        for i, val in enumerate(case_values):
            const = val[CONSTANT] if len(val) == 1 and CONSTANT in val else None
            child = case_children[i]
            if val == 'default':
                def_child = child
            elif const is None:
                if_values.append(val)
                if_children.append(child)
            else:
                const_values.append(const)
                const_children.append(child)

        # first get nested if tree for children that are not constant (and therefore cannot be indexed)
        root = tree = {}
        for i, val in enumerate(if_values):
            if False in tree:
                tree = tree[False]  # nest if
            # c = _combine_linear_functions(cond, _negate_linear_function(val))  # gets difference between planes
            tree['if'] = {'eq': (branch, val)}  # creates equality comparison branch
            tree[True] = if_children[i]
            tree[False] = {}

        # if no cases for constant conditionals, just return if tree
        if len(const_children) == 0:
            if False in tree:
                tree[False] = def_child
            else:
                root = def_child
            return root

        if False in tree:
            tree = tree[False]  # nest if

        # then get switch-case tree (if needed)
        # compares branch against const values (index table)
        tree['if'] = KeyedPlane(KeyedVector(branch), const_values, 0)
        for i, child in enumerate(const_children):
            tree[i] = child
        tree[None] = def_child
        return root

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
