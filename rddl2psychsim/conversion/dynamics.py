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

        def _is_self_dyn(dyn_expr):
            return _is_linear_function(dyn_expr) and len(dyn_expr) == 1 and key in dyn_expr and dyn_expr[key] == 1

        # tries to get actions responsible for feature update from expression
        expr = self._convert_expression(expression, param_map, dependencies)
        action_dynamics = self._extract_action_dynamics(expr)

        # for each sub-expression, set dynamics for feature as provided by an action (or the world itself)
        for action, dyn_expr in action_dynamics.items():
            if _is_self_dyn(dyn_expr):
                continue  # ignore dynamics would not change variable
            tree = self._get_dynamics_tree(key, dyn_expr)
            self.world.setDynamics(key, action, tree)
            logging.info(f'Set dynamics for feature "{key}" associated with "{action}" to:\n{tree}')

    def _extract_action_dynamics(self, expression: Dict) -> Dict[ActionSet or bool, Dict]:

        def _is_action_conditioned(expr):
            if 'action' in expr:
                _, action, _ = expr['action']
                return action

        dynamics = {True: expression}  # default: assign dynamics to world
        if 'if' not in expression:
            return dynamics
        if_expr = expression['if']

        # check if we can pull actions from the sub-expressions in the form "if (action) then dyn_expr"
        action = _is_action_conditioned(if_expr)
        if action is not None:
            # if single action, assign True branch dynamics to action and process False branch
            dynamics = {action: expression[True]}
            dynamics.update(self._extract_action_dynamics(expression[False]))
            return dynamics

        if 'logic_and' in if_expr:
            # if AND, only one action can be in the conjunction
            action_idx = -1
            action = None
            sub_exprs = if_expr['logic_and']
            for i, sub_expr in enumerate(sub_exprs):
                a = _is_action_conditioned(sub_expr)
                if a is not None:
                    if action_idx != -1:
                        action_idx = -1  # one agent can only perform max one action at a time
                        break
                    action_idx = i
                    action = a
            if action_idx == -1:  # expression is not action-conditioned
                return dynamics

            # if only one action in conjunction, remove it from conjunction and set its dynamics
            and_expr = tuple(sub_expr for i, sub_expr in enumerate(sub_exprs) if i != action_idx)
            dynamics = self._extract_action_dynamics(expression[False])
            return {action: {'if': {'logic_and': and_expr}, True: expression[True], False: dynamics[True]},
                    **dynamics}

        if 'logic_or' in if_expr:
            # if OR, multiple actions can be in the disjunction, will all have the same dynamics
            idxs = set()
            action_dynamics = []
            sub_exprs = if_expr['logic_or']
            for i, sub_expr in enumerate(sub_exprs):
                # tries to get action-conditioned dynamics for sub-expr
                dyn = self._extract_action_dynamics(
                    {'if': sub_expr, True: expression[True], False: expression[False]})
                for k in dyn.keys():
                    # gets action-conditioned dynamics only
                    if isinstance(k, ActionSet):
                        idxs.add(i)
                        action_dynamics.append({k: dyn[k]})
            if len(action_dynamics) == 0:
                # expression is not action-conditioned
                return dynamics

            # set True branch dynamics to actions, remove them from disjunction and set dynamics to world
            or_expr = tuple(sub_expr for i, sub_expr in enumerate(sub_exprs) if i not in idxs)
            dynamics = {}
            for dyn in action_dynamics:
                dynamics.update(dyn)  # add action-conditioned dynamics
            if len(or_expr) > 0:
                # set remaining "if" dynamics to world
                dynamics[True] = {'if': {'logic_or': or_expr}, True: expression[True], False: expression[False]}
            else:
                dynamics.update(self._extract_action_dynamics(expression[False]))  # process False branch
            return dynamics

        return dynamics

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
            return self._get_decision_tree(
                self._get_if_tree(expr, {CONSTANT: True}, {CONSTANT: False}), leaf_func)

        if 'if' in expr and len(expr) > 1:
            # check if no plane (branch) provided, then expression's truth value has to be resolved
            if isinstance(expr['if'], dict) and len(expr['if']) == 1:
                return self._get_decision_tree(self._get_if_tree(expr['if'], expr[True], expr[False]), leaf_func)

            assert isinstance(expr['if'], KeyedPlane), \
                f'Could not parse RDDL expression, got invalid branch: "{expr["if"]}"!'

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

    def _get_if_tree(self, branch: Dict, true_child: Dict, false_child: Dict) -> Dict:

        # first check if condition is constant, return child accordingly
        const_val = _get_const_val(branch, bool)
        if const_val is not None:
            return true_child if const_val else false_child

        # then tries to get branching plane directly from condition
        plane, negate = self._get_plane(branch)
        if plane is not None:
            # if we have a valid plane, then simply return if tree
            return {'if': plane,
                    True: false_child if negate else true_child,
                    False: true_child if negate else false_child}

        # otherwise we have to build (possibly nested) PWL trees
        if 'logic_and' in branch:
            # negate to have disjunction, ~(A ^ B ^ ...) <= > ~A | ~B | ...
            return self._get_if_tree({'logic_or': [{'not': sub_expr} for sub_expr in branch['logic_and']]},
                                     false_child,
                                     true_child)

        if 'logic_or' in branch:
            sub_exprs = branch['logic_or']

            # check expression in the form [ (VAR == CONST1 ^ ...) | (VAR == CONST2 ^ ...) | ...]
            # e.g., when we had a RDDL expression in the form exists_{?x : obj}[ VAR == ?x ^ ...]
            conds = {}
            str_to_conds = {}
            cond_idxs = []
            for i, sub_expr in enumerate(sub_exprs):
                and_exprs = sub_expr['logic_and'] if 'logic_and' in sub_expr else [sub_expr]
                for j, and_expr in enumerate(and_exprs):
                    if 'eq' in and_expr:
                        lhs = and_expr['eq'][0]
                        rhs = and_expr['eq'][1]
                        cond = None
                        c_val = None
                        if _is_linear_function(lhs):
                            c_val = _get_const_val(rhs)
                            if c_val is not None:
                                cond = lhs
                        elif _is_linear_function(rhs):
                            c_val = _get_const_val(lhs)
                            if c_val is not None:
                                cond = rhs
                        if cond is not None and c_val is not None:
                            if str(cond) not in str_to_conds:
                                str_to_conds[str(cond)] = cond
                                conds[str(cond)] = []
                            conds[str(cond)].append((c_val, and_exprs[:j] + and_exprs[j + 1:]))
                            cond_idxs.append(i)

            # transform sub expressions that have more than one possible value ofr a condition into a "switch"
            sub_exprs = [sub_exprs[i] for i in range(len(sub_exprs)) if i not in cond_idxs]
            for cond in conds.keys():
                case_values = ['default']
                case_exprs = [{CONSTANT: False}]
                for c_val, and_exprs in conds[cond]:
                    case_values.append({CONSTANT: c_val})
                    case_exprs.append({'logic_and': and_exprs} if len(and_exprs) > 0 else {CONSTANT: True})
                if len(case_values) > 2:
                    sub_exprs.append({'switch': (str_to_conds[cond], case_values, case_exprs)})  # add switch
                else:
                    # if only one value to compare against, then revert to VAR == CONST formulation, ie no switch
                    and_exprs = case_exprs[1]['logic_and'] if 'logic_and' in case_exprs[1] else []
                    sub_exprs.append({'logic_and': [{'eq': (str_to_conds[cond], case_values[1])}, *and_exprs]})

            if len(sub_exprs) == 1:  # check OR of only one element
                return self._get_if_tree(sub_exprs[0], true_child, false_child)

            # otherwise builds nested OR tree
            lhs = sub_exprs[0]
            rhs = {'logic_or': sub_exprs[1:]} if len(sub_exprs) > 2 else sub_exprs[1]
            return self._get_if_tree(lhs,
                                     true_child,
                                     self._get_if_tree(rhs, true_child, false_child))

        if 'not' in branch:
            # if NOT, just swap children
            return self._get_if_tree(branch['not'], false_child, true_child)

        if 'equiv' in branch:
            # if logical EQUIVALENCE, true iff both sides are true or both are false
            lhs, rhs = branch['equiv']
            return self._get_if_tree(lhs,
                                     self._get_if_tree(rhs, true_child, false_child),
                                     self._get_if_tree(rhs, false_child, true_child))

        if 'imply' in branch:
            # if IMPLICATION, false only if left is true and right is false
            lhs, rhs = branch['imply']
            return self._get_if_tree(lhs,
                                     self._get_if_tree(rhs, true_child, false_child),
                                     true_child)

        if 'switch' in branch:
            # if SWITCH, then propagate "if" children at each case child
            switch_branch, case_values, case_children = branch['switch']
            case_if_children = []
            for case_child in case_children:  # case child used as branch for the if subtree
                case_if_children.append({'if': case_child, True: true_child, False: false_child})
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

    def _get_plane(self, expr: Dict, negate: bool = False) -> Tuple[KeyedPlane or None, bool]:
        # signature: KeyedPlane(KeyedVector(weights), threshold, comparison)

        if 'not' in expr:
            # if NOT, get negated operation
            return self._get_plane(expr['not'], not negate)

        if _is_linear_function(expr):
            # assumes linear combination of all features in vector has to be > 0.5,
            # which is truth value in PsychSim (see psychsim.world.World.float2value)
            if negate:
                expr = _negate_linear_function(expr)
            return KeyedPlane(KeyedVector(expr), 0.5 if any(v > 0 for v in expr.values()) else -0.5, 1), False

        if 'action' in expr and len(expr['action']) == 3:
            # conditional on specific agent's action (agent's action == the action)
            agent, action, future = expr['action']
            key = makeFuture(actionKey(agent.name)) if future else actionKey(agent.name)  # check future vs prev action
            if negate:
                return (KeyedPlane(KeyedVector({key: 1}), action, 1) |
                        KeyedPlane(KeyedVector({key: 1}), action, -1)), False
            else:
                return KeyedPlane(KeyedVector({key: 1}), action, 0), False

        if 'linear_and' in expr and len(expr) == 1:
            # AND of features (sum > w_sum - 0.5), see psychsim.world.World.float2value
            # for negation, ~(A ^ B ^ ...) <=> ~A | ~B | ...
            expr = expr['linear_and']
            if negate:
                return self._get_plane({'linear_or': _negate_linear_function(expr)}, False)
            else:
                return KeyedPlane(KeyedVector(expr), sum([v for v in expr.values() if v > 0]) - 0.5, 1), False

        if 'linear_or' in expr and len(expr) == 1:
            # OR of features (A | B | ...) <=> ~(~A ^ ~B ^ ...)
            # for negation, ~(A | B | ...) <=> ~A ^ ~B ^ ...
            expr = expr['linear_or']
            if negate:
                return self._get_plane({'linear_and': _negate_linear_function(expr)}, False)
            else:
                expr = _negate_linear_function(expr)
                return KeyedPlane(KeyedVector(expr), sum([v for v in expr.values() if v > 0]) - 0.5 + EPS, -1), False

        if 'logic_and' in expr and len(expr) == 1:
            # negate to try to have disjunction: ~(A ^ B ^ ...) <= > ~A | ~B | ...
            return self._get_plane({'logic_or': [{'not': sub_expr} for sub_expr in expr['logic_and']]}, not negate)

        if 'logic_or' in expr and len(expr) == 1:
            if len(expr['logic_or']) == 1:  # check disjunction of only one element
                return self._get_plane(expr['logic_or'][0], negate)

            # get disjunction between sub-expressions (all planes must be valid)
            # for negation, ~(A | B | ...) <=> ~A ^ ~B ^ ...
            sub_exprs = expr['logic_or']
            planes = [self._get_plane(sub_expr) for sub_expr in sub_exprs]
            negates = [p[1] for p in planes]
            planes = [p[0] for p in planes]
            if None in planes:
                return None, negate
            if all(negates) and all(len(plane.planes) == 1 or plane.isConjunction for plane in planes):
                # ~A | ~B | ... <=> ~ (A ^ B ^ ...)
                return reduce(lambda p1, p2: p1 & p2, planes), not negate  # conjunction
            invalid = any(negates) or any(len(plane.planes) > 1 and plane.isConjunction for plane in planes)
            return (None if invalid else reduce(lambda p1, p2: p1 | p2, planes)), negate  # disjunction

        # test binary operators
        op = next(iter(expr.keys()))
        if len(expr) == 1 and op in {'eq', 'neq', 'gt', 'lt', 'geq', 'leq', 'imply'}:
            lhs, rhs = expr[op]
            if 'eq' in expr:
                if negate:
                    return self._get_plane({'neq': expr['eq']}, False)
                else:
                    # takes equality of pwl comb in vectors (difference==0 or expr==threshold)
                    weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                    return KeyedPlane(KeyedVector(weights), thresh, 0), False

            if 'neq' in expr:
                if negate:
                    return self._get_plane({'eq': expr['neq']}, False)
                else:
                    # takes inequality of pwl comb in vectors (difference!=0 or expr!=threshold)
                    weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                    return (KeyedPlane(KeyedVector(weights), thresh, 1) |
                            KeyedPlane(KeyedVector(weights), thresh, -1)), False

            if 'gt' in expr:
                if negate:
                    return self._get_plane({'leq': expr['gt']}, False)
                else:
                    # takes diff of pwl comb in vectors (difference>0 or expr>threshold)
                    weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                    return KeyedPlane(KeyedVector(weights), thresh, 1), False

            if 'lt' in expr:
                if negate:
                    return self._get_plane({'geq': expr['lt']}, False)
                else:
                    # takes diff of pwl comb in vectors (difference<0 or expr<threshold)
                    weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                    return KeyedPlane(KeyedVector(weights), thresh, -1), False

            if 'geq' in expr:
                if negate:
                    return self._get_plane({'lt': expr['geq']}, False)
                else:
                    # takes diff of pwl comb in vectors (difference>=0 or expr>=threshold)
                    weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                    if isinstance(thresh, str):
                        return (KeyedPlane(KeyedVector(weights), thresh, 1) |
                                KeyedPlane(KeyedVector(weights), thresh, 0)), False
                    else:
                        return KeyedPlane(KeyedVector(weights), thresh - EPS, 1), False

            if 'leq' in expr:
                if negate:
                    return self._get_plane({'gt': expr['leq']}, False)
                else:
                    # takes diff of pwl comb in vectors (difference<=0 or expr<=threshold)
                    weights, thresh = self._get_relational_plane_thresh(lhs, rhs)
                    if isinstance(thresh, str):
                        return (KeyedPlane(KeyedVector(weights), thresh, -1) |
                                KeyedPlane(KeyedVector(weights), thresh, 0)), False
                    else:
                        return KeyedPlane(KeyedVector(weights), thresh + EPS, -1), False

            if 'imply' in expr:
                # if IMPLICATION, false only if left is true and right is false, ie
                # true if left is false or right is true
                if not (_is_linear_function(lhs) and _is_linear_function(rhs)):
                    return None, negate  # both operands have to be linear combinations
                if negate:
                    return (KeyedPlane(KeyedVector(lhs), 0.5, 1) & KeyedPlane(KeyedVector(rhs), 0.5 + EPS, -1)), False
                else:
                    return (KeyedPlane(KeyedVector(lhs), 0.5 + EPS, -1) | KeyedPlane(KeyedVector(rhs), 0.5, 1)), False

        return None, negate

    def _get_relational_plane_thresh(self, lhs: Dict, rhs: Dict) -> Tuple[Dict, str or int]:
        if _is_linear_function(lhs) and _is_linear_function(rhs):
            op = _combine_linear_functions(lhs, _negate_linear_function(rhs))
            return op, 0  # if both sides are planes, returns difference (threshold 0)

        if _is_linear_function(lhs) and self._is_constant_expr(rhs):
            return lhs, rhs[CONSTANT]  # if comparison with enum, return enum value as threshold

        if self._is_constant_expr(lhs) and _is_linear_function(rhs):
            return rhs, lhs[CONSTANT]  # if comparison with enum, return enum value as threshold

        raise ValueError(f'Cannot parse expression, non-PWL relational operation between {lhs} and {rhs}!')
