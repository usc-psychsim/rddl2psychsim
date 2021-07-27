import math
import itertools as it
import numpy as np
from typing import Dict, Union, List, Set, Any
from pyrddl.expr import Expression
from psychsim.pwl import CONSTANT, makeFuture
from rddl2psychsim.conversion import _ConverterBase
from rddl2psychsim.rddl import expression_to_rddl

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def _is_linear_function(expr: Dict) -> bool:
    """
    Checks whether the given expression represents a linear function, i.e., a set of named-weights.
    :param Dict expr: the dictionary representing the (accumulated) expression.
    :rtype: bool
    :return: `True` if the dictionary contains string keys and the values are of type `float`, `int` or `bool`.
    Returns `False` otherwise.
    """
    return isinstance(expr, dict) and \
           all(isinstance(k, str) for k in expr.keys()) and \
           all(type(v) in [float, int, bool] for v in expr.values())


def _combine_linear_functions(expr1: Dict, expr2: Dict) -> Dict:
    """
    Combines the given linear functions into a single named-vector dictionary. If a key is present in both functions,
    then its new weight is the sum of weights in the original expressions. If a combined value is zero, then the
    corresponding key is removed from the returned function.
    :param Dict expr1: the dictionary representing one expression.
    :param Dict expr2: the dictionary representing the other expression.
    :rtype: Dict
    :return: a dictionary containing a set of named weights resulting from the combination of linear functions.
    """
    assert _is_linear_function(expr1) and _is_linear_function(expr2), \
        f'Could not parse expression, invalid linear operation in "{expr1}" or "{expr2}"!'
    expr1 = dict(expr1)  # copy
    for k, v in expr2.items():
        expr1[k] = expr1[k] + v if k in expr1 else v  # add weight if key already in dict
        if expr1[k] == 0:
            del expr1[k]  # remove if weight is 0
    return {CONSTANT: 0} if len(expr1) == 0 else expr1


def _negate_linear_function(expr):
    """
    Negates the weights associated with all keys in the given linear function.
    :param Dict expr: the dictionary representing the (accumulated) expression.
    :rtype: Dict
    :return: a new dictionary whose values for each key are the negation of the original values.
    """
    assert _is_linear_function(expr), f'Could not parse expression, invalid linear operation in "{expr}"!'
    return {k: -v for k, v in expr.items()}  # just negate the weight


def _scale_linear_function(expr, factor):
    """
    Multiplies the weights associated with all keys in the given linear function by a given value.
    Equivalent to a scaling function.
    :param Dict expr: the dictionary representing the (accumulated) expression.
    :param float factor: the multiplication factor.
    :rtype: Dict
    :return: a new dictionary whose values for each key are the original values multiplied by the given factor.
    """
    assert _is_linear_function(expr), f'Could not parse expression, invalid linear operation in "{expr}"!'
    return {k: v * factor for k, v in expr.items()}  # just scale the weight by the constant factor


def _get_const_val(expr: Dict, c_type: type = None) -> Union[float, None]:
    """
    Tests whether the given expression corresponds to a constant function and return it's value.
    :param Dict expr: the dictionary representing the (accumulated) expression.
    :param expr: the dictionary representing the (accumulated) expression.
    :param type c_type: the type of constant we are expecting from the expression.
    :rtype: float or None
    :return: the constant value of the expression or `None` if the expression is not a constant.
    """
    try:
        if not isinstance(expr, dict):
            return None
        if len(expr) == 1 and CONSTANT in expr:
            return c_type(expr[CONSTANT]) if c_type is not None else expr[CONSTANT]
        if len(expr) == 1 and next(iter(expr.values())) == 0:
            return c_type(0) if c_type is not None else 0
        return None
    except ValueError:
        return None


def _propagate_switch_expr(op: str, *args: Dict[str, Any]) -> Dict:
    """
    Propagates a binary expression to the leaves of a switch expression.
    For example, "switch(...){case a: expr_a, ... } > 1" is transformed to "switch(...){case a: expr_a > 1, ...}"
    :param str op: the outer operation to be propagated to the switch case children expressions.
    :param dict[str] args: the arguments for the outer operation, one of which is a switch expression.
    :rtype: dict
    :return: a switch expression in which the outer operation was propagated through its leaf (child) expressions.
    """
    switch_arg = None
    switch_idx = -1
    new_args = []
    for i, arg in enumerate(args):
        if 'switch' in arg:
            if switch_arg is not None:
                # if we already have a switch arg, then choose "longest" one first (smaller resulting tree)
                if len(arg['switch'][1]) <= len(switch_arg['switch'][1]):
                    new_args.append(arg)  # ignore this switch if "shorter" than existing one
                    continue
                else:
                    new_args[switch_idx] = args[switch_idx]  # replace with "longer" switch
            switch_arg = arg
            switch_idx = i
            new_args.append(None)  # placeholder to replace child
        else:
            new_args.append(arg)
    if switch_arg is None:
        return {op: tuple(args)}
    switch_arg, case_values, case_children = switch_arg['switch']
    new_case_children = []
    for case_child in case_children:  # case child used as branch for the if subtree
        op_args = new_args.copy()
        op_args[switch_idx] = case_child
        new_case_children.append({op: tuple(op_args) if len(op_args) > 1 else op_args[0]})
    return {'switch': (switch_arg, case_values, new_case_children)}


class _ExpressionConverter(_ConverterBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _is_constant_expr(self, expr: Dict) -> bool:
        """
        Checks whether the given expression represents an enumerated (constant) value.
        :param Dict expr: the dictionary representing the expression.
        :rtype: bool
        :return: `True` if the dictionary contains a single element whose key equals PsychSim's `CONSTANT` and the value
        belongs to the set of known enumerated types for the domain.
        Returns `False` otherwise.
        """
        return isinstance(expr, dict) and len(expr) == 1 and CONSTANT in expr and \
               (self._is_enum_value(expr[CONSTANT]) or self._is_constant_value(expr[CONSTANT]))

    def _get_param_mappings(self, expression: Expression) -> List[Dict[str, str]]:
        # get mappings in the form param -> param type value for each param combination
        param_names = [arg[1][0] for arg in expression.args[:-1]]
        param_types = [arg[1][1] for arg in expression.args[:-1]]
        param_combs = self._get_all_param_combs(param_types)
        return [dict(zip(param_names, param_comb)) for param_comb in param_combs]

    def _convert_expression(self, expression: Expression,
                            param_map: Dict[str, str] = None,
                            dependencies: Set[str] = None) -> Dict:

        # process leaf node, try to get feature name or constant value
        e_type = expression.etype[0]
        args = expression.args
        if e_type == 'constant':
            return {CONSTANT: args}

        if e_type == 'penum':
            # just check if enumerated type is known
            val = args.replace('@', '')
            if self._is_enum_value(val):
                return {CONSTANT: val}
            raise ValueError(f'Could not find enumerated type from RDDL expression "{expression_to_rddl(expression)}"!')

        if e_type == 'param':
            # just check if parameter is known in the current scope and return its value
            if args in param_map:
                return {CONSTANT: param_map[args]}
            raise ValueError(f'Could not find parameter "{args}" in the current scope from '
                             f'RDDL expression "{expression_to_rddl(expression)}"!')

        if e_type == 'pvar':
            return self._convert_variable_expr(expression, param_map, dependencies)

        if e_type == 'arithmetic':
            return self._convert_arithmetic_expr(expression, param_map, dependencies)

        if e_type == 'boolean':
            return self._convert_boolean_expr(expression, param_map, dependencies)

        if e_type == 'relational':
            return self._convert_relational_expr(expression, param_map, dependencies)

        if e_type == 'control':
            return self._convert_control_expr(expression, param_map, dependencies)

        if e_type == 'randomvar':
            return self._convert_distribution_expr(expression, param_map, dependencies)

        if e_type == 'aggregation':
            return self._convert_aggregation_expr(expression, param_map, dependencies)

        # not yet implemented
        raise NotImplementedError(f'Cannot parse expression: "{expression_to_rddl(expression)}", '
                                  f'cannot handle type "{e_type}"!')

    def _convert_variable_expr(self, expression: Expression, param_map: Dict[str, str] = None,
                               dependencies: Set[str] = None) -> Dict:

        name, params = expression.args
        if params is not None:
            # processes variable's parameters
            param_vals = []
            feat_idxs = {}  # stores indexes of parameters that are variables
            for i, p in enumerate(params):
                if param_map is not None and p in param_map:
                    param_vals.append([param_map[p]])  # replace param placeholder with value on dict
                elif isinstance(p, str) and self._is_enum_value(p):
                    param_vals.append([p.replace('@', '')])  # replace with enum value
                elif isinstance(p, Expression):
                    # param is a variable expression, so convert and check its type
                    p = self._convert_expression(p, param_map, dependencies)
                    assert len(p) == 1, f'Parameter is not a constant or variable name: "{p}"' \
                                        f'in RDDL expression "{expression_to_rddl(expression)}"!'
                    feat_name = next(iter(p.keys()))
                    if feat_name == CONSTANT:
                        param_vals.append([p[CONSTANT]])  # if it's a constant, param equals its value
                    elif self.world.variables[feat_name]['domain'] == list:
                        # if it's a variable and has finite domain (enum or object), store type values
                        feat_idxs[feat_name] = i
                        param_vals.append(self.world.variables[feat_name]['elements'])
                    else:
                        ValueError(f'Unknown or infinite domain param {p} '
                                   f'in RDDL expression "{expression_to_rddl(expression)}"!')
                else:
                    raise ValueError(f'Unknown param {p} in RDDL expression "{expression_to_rddl(expression)}"!')

            # get combinations between parameter values
            param_combs = list(it.product(*param_vals))
            if len(param_combs) == 1:
                param_vals = tuple(param_combs[0])  # if only one parameter combination, move on
            else:
                # otherwise, create one (possibly nested) switch statement to contemplate all possible param values
                feat_idxs = sorted([(feat, idx) for feat, idx in feat_idxs.items() if len(param_vals[idx]) > 1],
                                   key=lambda feat_idx: len(param_vals[feat_idx[1]]))
                feats_case_vals = {}
                for param_comb in param_combs:
                    param_map.update(dict(zip(params, param_comb)))  # update map to replace variables with values
                    expr = self._convert_expression(expression, param_map, dependencies)
                    comb_key = tuple(param_comb[idx] for _, idx in feat_idxs)
                    feats_case_vals[comb_key] = expr  # store value/expression for combination

                def _create_nested_switch(cur_feat_idx, cur_comb):
                    if cur_feat_idx == len(feat_idxs):
                        # terminal case, just return value/expression for parameter combination
                        return feats_case_vals[cur_comb]

                    # otherwise create case-branch for each feature value
                    feat, idx = feat_idxs[cur_feat_idx]
                    case_vals = param_vals[idx]
                    case_branches = []
                    for feat_val in case_vals:
                        case_branches.append(_create_nested_switch(cur_feat_idx + 1, cur_comb + (feat_val,)))
                    case_vals = [{CONSTANT: v} for v in case_vals]
                    case_vals[-1] = 'default'  # replace last value with "default" case option
                    cond = {feat: 1}
                    return {'switch': (cond, case_vals, case_branches)}  # return a switch expression

                return _create_nested_switch(0, ())
        else:
            param_vals = (None,)
        f_name = (name,) + param_vals

        # check if it's a named constant, return it's value
        if self._is_constant(f_name):
            value = self._get_constant_value(f_name)
            return {CONSTANT: value}

        # check if we should get future (current) or old value, from dependency list and from name
        future = '\'' in name or (dependencies is not None and name in dependencies)

        # check if this variable refers to a known feature, return the feature
        if self._is_feature(f_name):  #
            f_name = self._get_feature(f_name)
            return {makeFuture(f_name) if future else f_name: 1.}

        # check if it's an action
        ag_actions = []
        for agent in self.world.agents.values():
            if self._is_action(f_name, agent):  # check if this variable refers to an agent's action
                ag_actions.append((agent, self._get_action(f_name, agent), future))
        if len(ag_actions) > 0:
            # TODO can do plane disjunction when supported in PsychSim
            # creates OR nested tree for matching any agents' actions
            or_tree = {'action': ag_actions[0]}
            for ag_action in ag_actions[1:]:
                or_tree = {'logical_or': (or_tree, {'action': ag_action})}
            return or_tree

        raise ValueError(f'Could not find feature, action or constant from RDDL expression '
                         f'"{expression_to_rddl(expression)}"!')

    def _convert_arithmetic_expr(self, expression: Expression, param_map: Dict[str, str] = None,
                                 dependencies: Set[str] = None) -> Dict:

        lhs = self._convert_expression(expression.args[0], param_map, dependencies)
        rhs = self._convert_expression(expression.args[1], param_map, dependencies) if len(expression.args) > 1 else {}
        lhs_const = _get_const_val(lhs, float)
        rhs_const = _get_const_val(rhs, float) if rhs is not None else None
        all_consts = lhs_const is not None and rhs_const is not None
        lhs_switch = 'switch' in lhs
        rhs_switch = 'switch' in rhs

        a_type = expression.etype[1]
        if a_type == '+':
            if all_consts:
                return {CONSTANT: lhs_const + rhs_const}  # reduce
            # if addition, just combine everything from both sides
            return _combine_linear_functions(lhs, rhs)

        if a_type == '-':
            # check equal sides, return 0
            if lhs == rhs:
                return {CONSTANT: 0}
            # check all constants
            if all_consts:
                return {CONSTANT: lhs_const - rhs_const}  # reduce
            if len(rhs) == 0:
                rhs = lhs  # just switch if we only have one argument
                lhs = {}
            # if subtraction, multiply right-hand side by -1 and combine
            return _combine_linear_functions(lhs, _negate_linear_function(rhs))

        if a_type == '*':
            # if multiplication, only works if one or both sides are constants
            if all_consts:
                return {CONSTANT: lhs_const * rhs_const}  # reduce
            if lhs_const is not None and _is_linear_function(rhs):
                return _scale_linear_function(rhs, lhs_const)  # multiply right-hand side by const
            if rhs_const is not None and _is_linear_function(lhs):
                return _scale_linear_function(lhs, rhs_const)  # multiply left-hand side by const
            raise ValueError(f'Non-PWL operation is not supported: "{expression_to_rddl(expression)}"!')

        elif a_type == '/':
            # check equal sides, return 1
            if lhs == rhs:
                return {CONSTANT: 1}
            # if division, only works if right or both sides are constants
            if all_consts:
                return {CONSTANT: lhs_const / rhs_const}  # reduce
            if rhs_const is not None and _is_linear_function(lhs):
                return _scale_linear_function(lhs, 1. / rhs_const)  # divide left-hand side by const
            raise ValueError(f'Non-PWL operation is not supported: "{expression_to_rddl(expression)}"!')

        raise NotImplementedError(f'Cannot parse arithmetic expression: "{expression_to_rddl(expression)}",'
                                  f'cannot handle type "{a_type}"!')

    def _convert_boolean_expr(self, expression: Expression, param_map: Dict[str, str] = None,
                              dependencies: Set[str] = None) -> Dict:

        lhs = self._convert_expression(expression.args[0], param_map, dependencies)
        rhs = self._convert_expression(expression.args[1], param_map, dependencies) if len(expression.args) > 1 else {}
        lhs_const = _get_const_val(lhs, bool)
        rhs_const = _get_const_val(rhs, bool)
        all_consts = lhs_const is not None and rhs_const is not None
        lhs_switch = 'switch' in lhs
        rhs_switch = 'switch' in rhs

        b_type = expression.etype[1]
        if b_type == '^':
            # if AND, both sides have to be True
            if all_consts:
                return {CONSTANT: rhs_const and lhs_const}
            if rhs_const is not None:
                return lhs if rhs_const else {CONSTANT: False}
            if lhs_const is not None:
                return rhs if lhs_const else {CONSTANT: False}
            # check equal sides, return one of them
            if lhs == rhs:
                return lhs
            orig_lhs = lhs
            if 'linear_and' in lhs and len(lhs) == 1:
                lhs = lhs['linear_and']  # tries to combine several AND together
            elif 'not' in lhs and len(lhs) == 1 and _is_linear_function(lhs['not']):
                lhs = _negate_linear_function(lhs['not'])  # tries to combine AND with NOT
            orig_rhs = rhs
            if 'linear_and' in rhs and len(rhs) == 1:
                rhs = rhs['linear_and']  # tries to combine several AND together
            elif 'not' in rhs and len(rhs) == 1 and _is_linear_function(rhs['not']):
                rhs = _negate_linear_function(rhs['not'])  # tries to combine AND with NOT
            if _is_linear_function(lhs) and _is_linear_function(rhs):
                # if both linear ops, just add everything from both sides (thresh >= len)
                return {'linear_and': _combine_linear_functions(lhs, rhs)}
            if lhs_switch or rhs_switch:
                return _propagate_switch_expr('logic_and', lhs, rhs)  # propagate expr switch children
            return {'logic_and': (orig_lhs, orig_rhs)}  # AND tree

        if b_type == '|':
            # if OR, one side has to be True
            if all_consts:
                return {CONSTANT: rhs_const or lhs_const}
            if rhs_const is not None:
                return {CONSTANT: True} if rhs_const else lhs
            if lhs_const is not None:
                return {CONSTANT: True} if lhs_const else rhs
            # check equal sides, return one of them
            if lhs == rhs:
                return lhs
            orig_lhs = lhs
            if 'linear_or' in lhs and len(lhs) == 1:
                lhs = lhs['linear_or']  # tries to combine several OR together
            elif 'not' in lhs and len(lhs) == 1 and _is_linear_function(lhs['not']):
                lhs = _negate_linear_function(lhs['not'])  # tries to combine OR with NOT
            orig_rhs = rhs
            if 'linear_or' in rhs and len(rhs) == 1:
                rhs = rhs['linear_or']  # tries to combine several OR together
            elif 'not' in rhs and len(rhs) == 1 and _is_linear_function(rhs['not']):
                rhs = _negate_linear_function(rhs['not'])  # tries to combine OR with NOT
            if _is_linear_function(lhs) and _is_linear_function(rhs):
                # if both vectors, just add everything from both sides (thresh > 0)
                return {'linear_or': _combine_linear_functions(lhs, rhs)}
            if lhs_switch or rhs_switch:
                return _propagate_switch_expr('logic_or', lhs, rhs)  # propagate expr to switch children
            return {'logic_or': (orig_lhs, orig_rhs)}  # OR tree

        if b_type == '~':
            # NOT
            if lhs_const is not None:
                return {CONSTANT: False if lhs_const else True}  # gets constant's truth value
            if 'not' in lhs and len(lhs) == 1:
                return lhs['not']  # double negation
            if 'linear_and' in lhs and len(lhs) == 1:  # ~(A ^ B) <=> ~A | ~B
                return {'linear_or': _negate_linear_function(lhs['linear_and'])}
            if 'linear_or' in lhs and len(lhs) == 1:  # ~(A | B) <=> ~A ^ ~B
                return {'linear_and': _negate_linear_function(lhs['linear_or'])}
            if lhs_switch or rhs_switch:
                return _propagate_switch_expr('not', lhs)  # propagate expression to switch children
            return {'not': lhs}  # defer for later processing

        if b_type == '<=>':
            # if EQUIV, sides have to be of equal boolean value
            if all_consts:
                return {CONSTANT: rhs_const == lhs_const}  # equal booleans
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}
            if lhs_switch or rhs_switch:
                return _propagate_switch_expr('equiv', lhs, rhs)  # propagate expression to switch children
            return {'equiv': (lhs, rhs)}  # defer for later processing

        if b_type == '=>':
            # if IMPLICATION, false only if left is true and right is false
            if all_consts:
                return {CONSTANT: rhs_const or not lhs_const}
            # check equal sides, always True
            if lhs == rhs:
                return {CONSTANT: True}
            if lhs_const is not None:
                if not lhs_const:
                    return {CONSTANT: True}  # left is false, so implication is true
                return rhs  # left is true, so right has to be true
            if rhs_const is not None:
                if rhs_const:
                    return {CONSTANT: True}  # right is true, so implication is true
                return {'not': lhs}  # right is false, negate left
            if lhs_switch or rhs_switch:
                return _propagate_switch_expr('imply', lhs, rhs)  # propagate expression to switch children
            return {'imply': (lhs, rhs)}  # defer for later processing

        raise NotImplementedError(f'Cannot parse boolean expression: "{expression_to_rddl(expression)}",'
                                  f'cannot handle type "{b_type}"!')

    def _convert_relational_expr(self, expression: Expression, param_map: Dict[str, str] = None,
                                 dependencies: Set[str] = None) -> Dict:

        lhs = self._convert_expression(expression.args[0], param_map, dependencies)
        rhs = self._convert_expression(expression.args[1], param_map, dependencies) if len(expression.args) > 1 else {}
        lhs_const = _get_const_val(lhs)
        rhs_const = _get_const_val(rhs)
        all_consts = lhs_const is not None and rhs_const is not None
        lhs_switch = 'switch' in lhs
        rhs_switch = 'switch' in rhs

        b_type = expression.etype[1]
        if b_type == '==':
            # if EQUALS, sides have to be of equal value
            if all_consts:
                return {CONSTANT: lhs_const == rhs_const}
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}
            if lhs_switch or rhs_switch:
                return _propagate_switch_expr('eq', lhs, rhs)  # propagate expression to switch children

            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_constant_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_constant_expr(rhs)), \
                f'Could not parse relational expression "{expression_to_rddl(expression)}", ' \
                f'invalid PWL equivalence composition!'
            return {'eq': (lhs, rhs)}  # defer for later processing

        if b_type == '~=':
            # if NOT_EQUAL, sides have to be of different value
            if all_consts:
                return {CONSTANT: lhs_const != rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}
            if lhs_switch or rhs_switch:
                return _propagate_switch_expr('neq', lhs, rhs)  # propagate expression to switch children

            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_constant_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_constant_expr(rhs)), \
                f'Could not parse relational expression "{expression_to_rddl(expression)}", ' \
                f'invalid PWL equivalence composition!'
            return {'neq': (lhs, rhs)}  # defer for later processing

        if b_type == '>':
            # if GREATER_THAN, left has to be of higher value than right
            if all_consts:
                return {CONSTANT: lhs_const > rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}
            if lhs_switch or rhs_switch:
                return _propagate_switch_expr('gt', lhs, rhs)  # propagate expression to switch children

            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_constant_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_constant_expr(rhs)), \
                f'Could not parse relational expression "{expression_to_rddl(expression)}", ' \
                f'invalid PWL equivalence composition!'
            return {'gt': (lhs, rhs)}  # defer for later processing

        if b_type == '<':
            # if LESS_THAN, left has to be of lower value than right
            if all_consts:
                return {CONSTANT: lhs_const < rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}
            if lhs_switch or rhs_switch:
                return _propagate_switch_expr('lt', lhs, rhs)  # propagate expression to switch children

            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_constant_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_constant_expr(rhs)), \
                f'Could not parse relational expression "{expression_to_rddl(expression)}", ' \
                f'invalid PWL equivalence composition!'
            return {'lt': (lhs, rhs)}  # defer for later processing

        if b_type == '>=':
            # if GREATER_OR_EQUAL, left has to be of equal or higher value than right
            if all_consts:
                return {CONSTANT: lhs_const >= rhs_const}
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}
            if lhs_switch or rhs_switch:
                return _propagate_switch_expr('geq', lhs, rhs)  # propagate expression to switch children

            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_constant_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_constant_expr(rhs)), \
                f'Could not parse relational expression "{expression_to_rddl(expression)}", ' \
                f'invalid PWL equivalence composition!'
            return {'geq': (lhs, rhs)}  # defer for later processing

        if b_type == '<=':
            # if LESS_OR_EQUAL, left has to be of equal or lower value than right
            if all_consts:
                return {CONSTANT: lhs_const <= rhs_const}
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}
            if lhs_switch or rhs_switch:
                return _propagate_switch_expr('leq', lhs, rhs)  # propagate expression to switch children
            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_constant_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_constant_expr(rhs)), \
                f'Could not parse relational expression "{expression_to_rddl(expression)}",' \
                f'invalid PWL equivalence composition!'
            return {'leq': (lhs, rhs)}  # defer for later processing

        raise NotImplementedError(f'Cannot parse relational expression: "{expression_to_rddl(expression)}",'
                                  f'cannot handle type "{b_type}"!')

    def _convert_control_expr(self, expression: Expression, param_map: Dict[str, str] = None,
                              dependencies: Set[str] = None) -> Dict:
        c_type = expression.etype[1]
        if c_type == 'if':
            # get condition and branches
            cond = self._convert_expression(expression.args[0], param_map, dependencies)
            true_branch = self._convert_expression(expression.args[1], param_map, dependencies)
            false_branch = self._convert_expression(expression.args[2], param_map, dependencies)
            cond_const = _get_const_val(cond, bool)
            if cond_const is not None:  # if constant condition, then simply choose branch
                return true_branch if cond_const else false_branch
            return {'if': cond, True: true_branch, False: false_branch}

        if c_type == 'switch':
            assert len(expression.args) > 1, f'Cannot parse switch expression: "{expression_to_rddl(expression)}", ' \
                                             f'no cases provided!'

            # get expression for terminal condition, has to be PWL
            cond = self._convert_expression(expression.args[0], param_map, dependencies)
            assert _is_linear_function(cond), \
                f'Cannot parse switch expression: "{expression_to_rddl(expression)}", switch condition is not PWL!'

            # get expressions for each of the branches
            case_values = []
            case_branches = []
            for arg in expression.args[1:]:
                case_type = arg[0]
                if case_type == 'case':
                    val = self._convert_expression(arg[1][0], param_map, dependencies)
                    branch = self._convert_expression(arg[1][1], param_map, dependencies)
                elif case_type == 'default':
                    assert 'default' not in case_values, \
                        f'Cannot parse switch expression: "{expression_to_rddl(expression)}", ' \
                        f'default branch defined more than once!'
                    val = 'default'
                    branch = self._convert_expression(arg[1], param_map, dependencies)
                else:
                    raise ValueError(f'Cannot parse switch expression: "{expression_to_rddl(expression)}", '
                                     f'unknown case type: "{case_type}"!')

                assert val == 'default' or _is_linear_function(val) or self._is_constant_expr(val), \
                    f'Cannot parse switch expression: "{expression_to_rddl(expression)}", case condition is not PWL!'
                case_values.append(val)
                case_branches.append(branch)

            assert 'default' in case_values, \
                f'Cannot parse switch expression: "{expression_to_rddl(expression)}", missing default branch!'
            return {'switch': (cond, case_values, case_branches)}

        raise NotImplementedError(f'Cannot parse control expression: "{expression_to_rddl(expression)}",'
                                  f'cannot handle type "{c_type}"!')

    def _convert_distribution_expr(self, expression: Expression, param_map: Dict[str, str] = None,
                                   dependencies: Set[str] = None) -> Dict:
        d_type = expression.etype[1]
        if d_type == 'Bernoulli':
            arg = self._convert_expression(expression.args[0], param_map, dependencies)
            assert _get_const_val(arg) is not None, \
                f'Cannot parse stochastic expression: "{expression_to_rddl(expression)}", ' \
                f'non-constant probability: "{arg}"!'
            p = arg[CONSTANT]
            return {'distribution': [({CONSTANT: 1}, p), ({CONSTANT: 0}, 1 - p)]}

        if d_type == 'KronDelta':
            # return the argument itself, although result should be int? From the docs:
            # "places all probability mass on its discrete argument v, discrete sample is thus deterministic"
            return self._convert_expression(expression.args[0], param_map, dependencies)

        if d_type == 'DiracDelta':
            # return the argument itself. From the docs:
            # "places all probability mass on its continuous argument v, continuous sample is thus deterministic"
            return self._convert_expression(expression.args[0], param_map, dependencies)

        if d_type == 'Discrete':
            def _get_value(val):
                if isinstance(val_type, tuple) and val_type[0] == 'enum_type' and self._is_enum(val_type[1]):
                    if self._is_enum_value(val):
                        return val.replace('@', '')
                    raise ValueError(f'Cannot parse stochastic expression: "{expression_to_rddl(expression)}", '
                                     f'unknown enum value "{val}" for type "{val_type}"')
                raise ValueError(f'Cannot parse stochastic expression: "{expression_to_rddl(expression)}",'
                                 f'unknown type "{val_type}"')

            assert len(expression.args) > 1, \
                f'Cannot parse stochastic expression: "{expression_to_rddl(expression)}", ' \
                f'must provide at least one value!'

            # return a key-value pair for the discrete distribution; has to be constant probability value
            val_type = expression.args[0]
            dist = []
            for arg in expression.args[1:]:
                k = _get_value(arg[1][0])
                v = _get_const_val(self._convert_expression(arg[1][1], param_map, dependencies), float)
                assert v is not None, \
                    f'Cannot parse stochastic expression: "{expression_to_rddl(expression)}", ' \
                    f'non-constant probability: "{v}"!'
                if v > 0:
                    dist.append(({CONSTANT: k}, v))

            assert np.isclose(np.sum([v for _, v in dist]), 1), \
                f'Cannot parse stochastic expression: "{expression_to_rddl(expression)}", ' \
                f'probabilities have to sum to 1!'
            return dist[0][0] if len(dist) == 1 else {'distribution': dist}

        if d_type == 'Normal':
            # check params, have to be linear functions
            mu = self._convert_expression(expression.args[0], param_map, dependencies)
            std = self._convert_expression(expression.args[1], param_map, dependencies)  # this is stdev, not variance!
            assert _is_linear_function(mu) and _is_linear_function(std), \
                f'Cannot parse stochastic expression: "{expression_to_rddl(expression)}", ' \
                f'mean and std have to be linear functions!'
            dist = []
            for i, b in enumerate(self._normal_bins):
                k = _combine_linear_functions(_scale_linear_function(std, b), mu)  # sample val = mean + bin * std
                if len(k) == 0:
                    k = {CONSTANT: 0}  # might have been ignored
                if _get_const_val(k, float) == -1:
                    k = {CONSTANT: - 1 + 1e-16}  # hack, apparently hash(-1) in python evaluates to -2, so give it nudge
                v = self._normal_probs[i]
                dist.append((k, v))

            return {'distribution': dist}

        if d_type == 'Poisson':
            # check param, has to be linear function
            lamb = self._convert_expression(expression.args[0], param_map, dependencies)
            assert _is_linear_function(lamb), \
                f'Cannot parse stochastic expression: "{expression_to_rddl(expression)}", ' \
                f'lambda has to be linear function!'
            dist = []
            std = math.sqrt(self._poisson_exp_rate)  # lambda expected to be around expected rate
            for i, b in enumerate(self._normal_bins):
                k = _combine_linear_functions({CONSTANT: std * b}, lamb)  # sample val = lambda + bin * std
                if len(k) == 0:
                    k = {CONSTANT: 0}  # might have been ignored
                if _get_const_val(k, float) == -1:
                    k = {CONSTANT: - 1 + 1e-16}  # hack, apparently hash(-1) in python evaluates to -2, so give it nudge
                v = self._normal_probs[i]
                dist.append((k, v))

            return {'distribution': dist}

        raise NotImplementedError(f'Cannot parse stochastic expression: "{expression_to_rddl(expression)}",'
                                  f'cannot handle type "{d_type}"!')

    def _convert_aggregation_expr(self, expression: Expression, param_map: Dict[str, str] = None,
                                  dependencies: Set[str] = None) -> Dict:
        d_type = expression.etype[1]
        if param_map is None:
            param_map = {}

        if d_type == 'sum':
            assert all(len(arg) == 2 and arg[0] == 'typed_var' for arg in expression.args[:-1]), \
                f'Cannot parse aggregation expression: "{expression_to_rddl(expression)}", invalid summation arguments!'

            # combine linear functions resulting from param substitutions in sub-expressions
            param_maps = self._get_param_mappings(expression)
            lf = {}
            for p_map in param_maps:
                expr = self._convert_expression(expression.args[-1], {**param_map, **p_map}, dependencies)
                lf = _combine_linear_functions(lf, expr)  # sum linear functions
            return {CONSTANT: 0} if len(lf) == 0 else lf

        if d_type == 'prod':
            assert all(len(arg) == 2 and arg[0] == 'typed_var' for arg in expression.args[:-1]), \
                f'Cannot parse aggregation expression: "{expression_to_rddl(expression)}", invalid product arguments!'

            # then combine linear functions resulting from param substitutions in sub-expressions
            param_maps = self._get_param_mappings(expression)
            lf = {CONSTANT: 1.}
            for p_map in param_maps:
                expr = self._convert_expression(expression.args[-1], {**param_map, **p_map}, dependencies)
                const_val = _get_const_val(expr, float)
                assert const_val is not None, \
                    f'Cannot parse aggregation expression: "{expression_to_rddl(expression)}", ' \
                    f'non-constant expression: "{expr}"!'
                lf = _scale_linear_function(lf, const_val)  # product of constant values
            return {CONSTANT: 0} if len(lf) == 0 else lf

        if d_type == 'forall':
            assert all(len(arg) == 2 and arg[0] == 'typed_var' for arg in expression.args[:-1]), \
                f'Cannot parse aggregation expression: "{expression_to_rddl(expression)}", ' \
                f'invalid conjunction arguments!'

            # combine param substitutions in sub-expressions
            param_maps = self._get_param_mappings(expression)
            sub_exprs = []
            for p_map in param_maps:
                sub_exprs.append(self._convert_expression(expression.args[-1], {**param_map, **p_map}, dependencies))

            # filter sub-expressions
            filtered_sub_exprs = []
            for i, expr in enumerate(sub_exprs):
                const_val = _get_const_val(expr, bool)
                if const_val is None:
                    filtered_sub_exprs.append(expr)
                elif not const_val:  # check False constant, then forall is False
                    return {CONSTANT: False}
                # otherwise got True constant, then does not affect forall

            sub_exprs = filtered_sub_exprs
            if len(sub_exprs) == 0:  # if nothing to check, forall is True
                return {CONSTANT: True}
            if len(sub_exprs) == 1 or all(s == sub_exprs[0] for s in sub_exprs[1:]):
                # if only one expression in forall, then just return it
                return sub_exprs[0]

            # if all linear ops, just add everything (thresh >= len)
            if all(_is_linear_function(expr) for expr in sub_exprs):
                lf = {}
                for expr in sub_exprs:
                    lf = _combine_linear_functions(lf, expr)  # sum linear functions
                return {CONSTANT: False} if len(lf) == 0 else {'linear_and': lf}

            # otherwise build AND plane conjunction
            return {'logic_and': sub_exprs}

        if d_type == 'exists':
            assert all(len(arg) == 2 and arg[0] == 'typed_var' for arg in expression.args[:-1]), \
                f'Cannot parse aggregation expression: "{expression_to_rddl(expression)}", ' \
                f'invalid disjunction arguments!'

            # combine param substitutions in sub-expressions
            param_maps = self._get_param_mappings(expression)
            sub_exprs = []
            for p_map in param_maps:
                sub_exprs.append(self._convert_expression(expression.args[-1], {**param_map, **p_map}, dependencies))

            # filter sub-expressions
            filtered_sub_exprs = []
            for i, expr in enumerate(sub_exprs):
                const_val = _get_const_val(expr, bool)
                if const_val is None:
                    filtered_sub_exprs.append(expr)
                elif const_val:  # check True constant, then exists is True
                    return {CONSTANT: True}
                # otherwise got False constant, then does not affect exists

            sub_exprs = filtered_sub_exprs
            if len(sub_exprs) == 0:  # if nothing to check, exists is False
                return {CONSTANT: False}
            if len(sub_exprs) == 1 or all(s == sub_exprs[0] for s in sub_exprs[1:]):
                # if only one expression in forall, then just return it
                return sub_exprs[0]

            # if all linear ops, just add everything (linear OR)
            if all(_is_linear_function(expr) for expr in sub_exprs):
                lf = {}
                for expr in sub_exprs:
                    lf = _combine_linear_functions(lf, expr)  # sum linear functions
                return {CONSTANT: False} if len(lf) == 0 else {'linear_or': lf}

            # otherwise build nested OR plane disjunction
            return {'logic_or': sub_exprs}

        raise NotImplementedError(f'Cannot parse aggregation expression: "{expression_to_rddl(expression)}",'
                                  f'cannot handle type "{d_type}"!')
