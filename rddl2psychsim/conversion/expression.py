import logging
import math
from typing import Dict, Tuple, Union, List
from pyrddl.expr import Expression
from psychsim.agent import Agent
from psychsim.pwl import CONSTANT, actionKey
from rddl2psychsim.conversion import _ConverterBase

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
           all(any(map(lambda t: isinstance(v, t), [float, int, bool])) for v in expr.values())


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
    expr1 = dict(expr1)  # copies
    for k, v in expr2.items():
        expr1[k] = expr1[k] + v if k in expr1 else v  # add weight if key already in dict
        if expr1[k] == 0:
            del expr1[k]  # remove if weight is 0
    return expr1


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


def _get_const_val(expr: Dict) -> Union[float, None]:
    """
    Tests whether the given expression corresponds to a constant function and return it's value.
    :param Dict expr: the dictionary representing the (accumulated) expression.
    :param expr: the dictionary representing the (accumulated) expression.
    :rtype: float or None
    :return: the constant value of the expression.
    """
    try:
        return None if not isinstance(expr, dict) else \
            float(expr[CONSTANT]) if len(expr) == 1 and CONSTANT in expr else None
    except ValueError:
        return None


class _ExpressionConverter(_ConverterBase):

    def __init__(self):
        super().__init__()

    def _is_enum_expr(self, expr: Dict) -> bool:
        """
        Checks whether the given expression represents an enumerated (constant) value.
        :param Dict expr: the dictionary representing the expression.
        :rtype: bool
        :return: `True` if the dictionary contains a single element whose key equals PsychSim's `CONSTANT` and the value
        belongs to the set of known enumerated types for the domain.
        Returns `False` otherwise.
        """
        return isinstance(expr, dict) and len(expr) == 1 and CONSTANT in expr and self._is_enum_type(expr[CONSTANT])

    def _get_relational_plane_thresh(self, lhs: Dict, rhs: Dict) -> Tuple[Dict, str or int]:
        if _is_linear_function(lhs) and _is_linear_function(rhs):
            op = _combine_linear_functions(lhs, _negate_linear_function(rhs))
            return op, 0  # if both sides are planes, returns difference (threshold 0)

        if _is_linear_function(lhs) and self._is_enum_expr(rhs):
            return lhs, rhs[CONSTANT]  # if comparison with enum, return enum value as threshold

        if self._is_enum_expr(lhs) and _is_linear_function(rhs):
            return rhs, lhs[CONSTANT]  # if comparison with enum, return enum value as threshold

        raise ValueError(f'Cannot parse expression, non-PWL relational operation between {lhs} and {rhs}!')

    def _get_pwl_tree(self, comp: Dict, true_branch: Dict, false_branch: Dict) -> Dict:
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

        if 'action' in comp and len(comp['action']) == 2:
            agent, action = comp['action']
            return {'if': ({actionKey(agent.name): 1.}, action, 0),  # conditional on specific agent's action
                    True: true_branch,
                    False: false_branch}

        if _is_linear_function(comp):
            # default: assumes linear combination of all features in vector has to be > 0.5,
            # which is truth value in PsychSim (see psychsim.world.World.float2value)
            return {'if': (comp, 0.5, 1),
                    True: true_branch,
                    False: false_branch}

        raise ValueError(f'Could not parse RDDL expression, unknown PWL tree comparison "{comp}"!')

    def _get_param_mappings(self, expression: Expression) -> List[Dict[str, str]]:
        # get mappings in the form param -> param type value for each param combination
        param_names = [arg[1][0] for arg in expression.args[:-1]]
        param_types = [arg[1][1] for arg in expression.args[:-1]]
        param_combs = self._get_all_param_combs(param_types)
        return [dict(zip(param_names, param_comb)) for param_comb in param_combs]

    def _convert_expression(self, expression: Expression, param_map: Dict[str, str] = None) -> Dict:

        # process leaf node, try to get feature name or constant value
        e_type = expression.etype[0]
        args = expression.args
        if e_type == 'constant':
            try:
                return {CONSTANT: args}
            except ValueError as e:
                logging.info(f'Could not convert value "{args}" to float in RDDL expression "{expression}"!')
                raise e

        if e_type == 'penum':
            # just check if enumerated type is known
            val = args.replace('@', '')
            if self._is_enum_type(val):
                return {CONSTANT: val}
            raise ValueError(f'Could not find enumerated type from RDDL expression "{expression}"!')

        if e_type == 'pvar':
            name, params = args
            if params is not None:
                # try to replace param placeholder with value on dict
                params = tuple([param_map[p] for p in params if p in param_map])
            else:
                params = (None,)
            name = (name,) + params

            if self._is_feature(name):  # feature
                return {self._get_feature(name): 1.}

            ag_actions = []
            for agent in self.world.agents.values():
                if self._is_action(name, agent):
                    ag_actions.append((agent, self._get_action(name, agent)))  # identify this an agent's action
            if len(ag_actions) > 0:
                # TODO can do plane disjunction when supported in PsychSim
                # creates OR nested tree for matching any agents' actions
                or_tree = {'action': ag_actions[0]}
                for ag_action in ag_actions[1:]:
                    or_tree = {'logical_or': (or_tree, {'action': ag_action})}
                return or_tree

            if self._is_constant(name):  # named constant
                try:
                    value = self._get_constant_value(name)
                    return {CONSTANT: float(value)}
                except ValueError as e:
                    logging.info(f'Could not convert value "{value}" to float in RDDL expression "{expression}"!')
                    raise e

            raise ValueError(f'Could not find feature, action or constant from RDDL expression "{expression}"!')

        if e_type == 'arithmetic':
            return self._convert_arithmetic_expr(expression, param_map)

        if e_type == 'boolean':
            return self._convert_boolean_expr(expression, param_map)

        if e_type == 'relational':
            return self._convert_relational_expr(expression, param_map)

        if e_type == 'control':
            return self._convert_control_expr(expression, param_map)

        if e_type == 'randomvar':
            return self._convert_distribution_expr(expression, param_map)

        if e_type == 'aggregation':
            return self._convert_aggregation_expr(expression, param_map)

        # not yet implemented
        raise NotImplementedError(f'Cannot parse expression: "{expression}" of type "{e_type}"!')

    def _convert_arithmetic_expr(self, expression: Expression, param_map: Dict[str, str] = None) -> Dict:

        lhs = self._convert_expression(expression.args[0], param_map)
        rhs = self._convert_expression(expression.args[1], param_map) if len(expression.args) > 1 else {}
        lhs_const = _get_const_val(lhs)
        rhs_const = _get_const_val(rhs) if rhs is not None else None
        all_consts = isinstance(lhs_const, float) and isinstance(rhs_const, float)
        a_type = expression.etype[1]

        if a_type == '+':
            if all_consts:
                return {CONSTANT: lhs_const + rhs_const}  # reduce
            # if addition, just combine everything from both sides
            return _combine_linear_functions(lhs, rhs)

        if a_type == '-':
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
            if isinstance(lhs_const, float) and _is_linear_function(rhs):
                return _scale_linear_function(rhs, lhs_const)  # multiply right-hand side by const
            if isinstance(rhs_const, float) and _is_linear_function(lhs):
                return _scale_linear_function(lhs, rhs_const)  # multiply left-hand side by const
            raise ValueError(f'Non-PWL operation is not supported: "{expression}"!')

        elif a_type == '/':
            # if division, only works if right or both sides are constants
            if all_consts:
                return {CONSTANT: lhs_const / rhs_const}  # reduce
            if isinstance(rhs_const, float) and _is_linear_function(lhs):
                return _scale_linear_function(lhs, 1. / rhs_const)  # divide left-hand side by const
            raise ValueError(f'Non-PWL operation is not supported: "{expression}"!')

        raise NotImplementedError(f'Cannot parse arithmetic expression: "{expression}" of type "{a_type}"!')

    def _convert_boolean_expr(self, expression: Expression, param_map: Dict[str, str] = None) -> Dict:

        lhs = self._convert_expression(expression.args[0], param_map)
        rhs = self._convert_expression(expression.args[1], param_map) if len(expression.args) > 1 else {}
        lhs_const = _get_const_val(lhs)
        rhs_const = _get_const_val(rhs)
        all_consts = isinstance(lhs_const, float) and isinstance(rhs_const, float)

        b_type = expression.etype[1]
        if b_type == '^':
            # if AND, both sides have to be True
            if all_consts:
                return {CONSTANT: float(bool(rhs_const) and bool(lhs_const))}
            if isinstance(rhs_const, float):
                return lhs if bool(rhs_const) else {CONSTANT: False}
            if isinstance(lhs_const, float):
                return rhs if bool(lhs_const) else {CONSTANT: False}
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
            return {'logic_and': (orig_lhs, orig_rhs)}  # AND tree

        if b_type == '|':
            # if OR, one side has to be True
            if all_consts:
                return {CONSTANT: bool(rhs_const) or bool(lhs_const)}
            if isinstance(rhs_const, float):
                return {CONSTANT: True} if bool(rhs_const) else lhs
            if isinstance(lhs_const, float):
                return {CONSTANT: True} if bool(lhs_const) else rhs
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
            return {'logic_or': (orig_lhs, orig_rhs)}  # OR tree

        if b_type == '~':
            # NOT
            if isinstance(lhs_const, float):
                return {CONSTANT: False if bool(lhs_const) else True}  # gets constant's truth value
            if 'not' in lhs and len(lhs) == 1:
                return lhs['not']  # double negation
            if 'linear_and' in lhs and len(lhs) == 1:  # ~(A ^ B) <=> ~A | ~B
                return {'linear_or': _negate_linear_function(lhs['linear_and'])}
            if 'linear_or' in lhs and len(lhs) == 1:  # ~(A | B) <=> ~A ^ ~B
                return {'linear_and': _negate_linear_function(lhs['linear_or'])}
            return {'not': lhs}  # defer for later processing

        if b_type == '<=>':
            # if EQUIV, sides have to be of equal boolean value
            if all_consts:
                return {CONSTANT: bool(rhs_const) == bool(lhs_const)}  # equal booleans
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}
            return {'equiv': (lhs, rhs)}  # defer for later processing

        if b_type == '=>':
            # if IMPLICATION, false only if left is true and right is false
            if all_consts:
                return {CONSTANT: bool(rhs_const) or not bool(lhs_const)}
            if isinstance(lhs_const, float):
                if not bool(lhs_const):
                    return {CONSTANT: True}  # left is false, so implication is true
                return rhs  # left is true, so right has to be true
            if isinstance(rhs_const, float):
                if bool(rhs_const):
                    return {CONSTANT: True}  # right is true, so implication is true
                return {'not': lhs}  # right is false, negate left
            return {'imply': (lhs, rhs)}  # defer for later processing

        raise NotImplementedError(f'Cannot parse boolean expression: "{expression}" of type "{b_type}"!')

    def _convert_relational_expr(self, expression: Expression, param_map: Dict[str, str] = None) -> Dict:

        lhs = self._convert_expression(expression.args[0], param_map)
        rhs = self._convert_expression(expression.args[1], param_map) if len(expression.args) > 1 else {}
        lhs_const = _get_const_val(lhs)
        rhs_const = _get_const_val(rhs)
        all_consts = isinstance(lhs_const, float) and isinstance(rhs_const, float)

        b_type = expression.etype[1]
        if b_type == '==':
            # if EQUALS, sides have to be of equal value
            if all_consts:
                return {CONSTANT: lhs_const == rhs_const}
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}

            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_enum_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_enum_expr(rhs)), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'eq': (lhs, rhs)}  # defer for later processing

        if b_type == '~=':
            # if NOT_EQUAL, sides have to be of different value
            if all_consts:
                return {CONSTANT: lhs_const != rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}

            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_enum_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_enum_expr(rhs)), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'neq': (lhs, rhs)}  # defer for later processing

        if b_type == '>':
            # if GREATER_THAN, left has to be of higher value than right
            if all_consts:
                return {CONSTANT: lhs_const > rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}

            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_enum_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_enum_expr(rhs)), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'gt': (lhs, rhs)}  # defer for later processing

        if b_type == '<':
            # if LESS_THAN, left has to be of lower value than right
            if all_consts:
                return {CONSTANT: lhs_const < rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}

            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_enum_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_enum_expr(rhs)), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'lt': (lhs, rhs)}  # defer for later processing

        if b_type == '>=':
            # if GREATER_OR_EQUAL, left has to be of equal or higher value than right
            if all_consts:
                return {CONSTANT: lhs_const >= rhs_const}
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}

            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_enum_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_enum_expr(rhs)), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'geq': (lhs, rhs)}  # defer for later processing

        if b_type == '<=':
            # if LESS_OR_EQUAL, left has to be of equal or lower value than right
            if all_consts:
                return {CONSTANT: lhs_const <= rhs_const}
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}

            # left and right have to be linear funcs
            assert (_is_linear_function(lhs) or self._is_enum_expr(lhs)) and \
                   (_is_linear_function(rhs) or self._is_enum_expr(rhs)), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'leq': (lhs, rhs)}  # defer for later processing

        raise NotImplementedError(f'Cannot parse relational expression: "{expression}" of type "{b_type}"!')

    def _convert_control_expr(self, expression: Expression, param_map: Dict[str, str] = None) -> Dict:
        c_type = expression.etype[1]
        if c_type == 'if':
            # get condition and branches
            cond = self._convert_expression(expression.args[0], param_map)
            true_branch = self._convert_expression(expression.args[1], param_map)
            false_branch = self._convert_expression(expression.args[2], param_map)
            return self._get_pwl_tree(cond, true_branch, false_branch)

        if c_type == 'switch':
            assert len(expression.args) > 1, f'Cannot parse switch expression: "{expression}", no cases provided!'

            # get expression for terminal condition, has to be PWL
            cond = self._convert_expression(expression.args[0], param_map)
            assert _is_linear_function(
                cond), f'Cannot parse switch expression: "{expression}", switch condition is not PWL!'

            # get expressions for each of the branches
            case_values = []
            case_branches = []
            for arg in expression.args[1:]:
                case_type = arg[0]
                if case_type == 'case':
                    val = self._convert_expression(arg[1][0], param_map)
                    branch = self._convert_expression(arg[1][1], param_map)
                elif case_type == 'default':
                    assert 'default' not in case_values, f'Cannot parse switch expression: "{expression}", ' \
                                                         f'default branch defined more than once!'
                    val = 'default'
                    branch = self._convert_expression(arg[1], param_map)
                else:
                    raise ValueError(f'Cannot parse switch expression: "{expression}", '
                                     f'unknown case type: "{case_type}"!')

                assert val == 'default' or _is_linear_function(val) or self._is_enum_expr(val), \
                    f'Cannot parse switch expression: "{expression}", case condition is not PWL!'
                case_values.append(val)
                case_branches.append(branch)

            assert 'default' in case_values, f'Cannot parse switch expression: "{expression}", missing default branch!'
            return {'switch': (cond, case_values, case_branches)}

        raise NotImplementedError(f'Cannot parse control expression: "{expression}" of type "{c_type}"!')

    def _convert_distribution_expr(self, expression: Expression, param_map: Dict[str, str] = None) -> Dict:
        d_type = expression.etype[1]
        if d_type == 'Bernoulli':
            arg = self._convert_expression(expression.args[0], param_map)
            assert _get_const_val(arg) is not None, \
                f'Cannot parse stochastic expression: "{expression}", non-constant probability: "{arg}"!'
            p = arg[CONSTANT]
            return {'distribution': [({CONSTANT: 1}, p), ({CONSTANT: 0}, 1 - p)]}

        if d_type == 'KronDelta':
            # return the argument itself, although result should be int? From the docs:
            # "places all probability mass on its discrete argument v, discrete sample is thus deterministic"
            return self._convert_expression(expression.args[0], param_map)

        if d_type == 'DiracDelta':
            # return the argument itself. From the docs:
            # "places all probability mass on its continuous argument v, continuous sample is thus deterministic"
            return self._convert_expression(expression.args[0], param_map)

        if d_type == 'Discrete':
            def _get_value(val):
                if isinstance(val_type, tuple) and val_type[0] == 'enum_type' and self._is_enum(val_type[1]):
                    if self._is_enum_type(val):
                        return val.replace('@', '')
                    raise ValueError(f'Cannot parse stochastic expression: "{expression}", '
                                     f'unknown enum value "{val}" for type "{val_type}"')
                raise ValueError(f'Cannot parse stochastic expression: "{expression}", unknown type "{val_type}"')

            assert len(expression.args) > 1, \
                f'Cannot parse stochastic expression: "{expression}", must provide at least one value!'

            # return a key-value pair for the discrete distribution; has to be constant probability value
            val_type = expression.args[0]
            dist = []
            for arg in expression.args[1:]:
                k = _get_value(arg[1][0])
                v = _get_const_val(self._convert_expression(arg[1][1], param_map))
                assert v is not None, \
                    f'Cannot parse stochastic expression: "{expression}", non-constant probability: "{v}"!'
                if v > 0:
                    dist.append(({CONSTANT: k}, v))

            assert sum(v for _, v in dist) == 1, \
                f'Cannot parse stochastic expression: "{expression}", probabilities have to sum to 1!'
            return {'distribution': dist}

        if d_type == 'Normal':
            # check params, have to be linear functions
            mu = self._convert_expression(expression.args[0], param_map)
            std = self._convert_expression(expression.args[1], param_map)  # assume this is stdev, not variance!
            assert _is_linear_function(mu) and _is_linear_function(std), \
                f'Cannot parse stochastic expression: "{expression}", mean and std have to be linear functions!'
            dist = []
            for i, b in enumerate(self._normal_bins):
                k = _combine_linear_functions(_scale_linear_function(std, b), mu)  # sample val = mean + bin * std
                if len(k) == 0:
                    k = {CONSTANT: 0}  # might have been ignored
                if _get_const_val(k) == -1:
                    k = {CONSTANT: - 1 + 1e-16}  # hack, apparently hash(-1) in python evaluates to -2, so give it nudge
                v = self._normal_probs[i]
                dist.append((k, v))

            return {'distribution': dist}

        if d_type == 'Poisson':
            # check param, has to be linear function
            lamb = self._convert_expression(expression.args[0], param_map)
            assert _is_linear_function(lamb), \
                f'Cannot parse stochastic expression: "{expression}", lambda has to be linear function!'
            dist = []
            std = math.sqrt(self._poisson_exp_rate)  # lambda expected to be around expected rate
            for i, b in enumerate(self._normal_bins):
                k = _combine_linear_functions({CONSTANT: std * b}, lamb)  # sample val = lambda + bin * std
                if len(k) == 0:
                    k = {CONSTANT: 0}  # might have been ignored
                if _get_const_val(k) == -1:
                    k = {CONSTANT: - 1 + 1e-16}  # hack, apparently hash(-1) in python evaluates to -2, so give it nudge
                v = self._normal_probs[i]
                dist.append((k, v))

            return {'distribution': dist}

        raise NotImplementedError(f'Cannot parse stochastic expression: "{expression}" of type "{d_type}"!')

    def _convert_aggregation_expr(self, expression: Expression, param_map: Dict[str, str] = None) -> Dict:
        d_type = expression.etype[1]
        if param_map is None:
            param_map = {}

        if d_type == 'sum':
            assert all(len(arg) == 2 and arg[0] == 'typed_var' for arg in expression.args[:-1]), \
                f'Cannot parse aggregation expression: "{expression}", invalid summation arguments!'

            # combine linear functions resulting from param substitutions in sub-expressions
            param_maps = self._get_param_mappings(expression)
            lf = {}
            for p_map in param_maps:
                expr = self._convert_expression(expression.args[-1], {**param_map, **p_map})
                lf = _combine_linear_functions(lf, expr)  # sum linear functions
            return {CONSTANT: 0} if len(lf) == 0 else lf

        if d_type == 'prod':
            assert all(len(arg) == 2 and arg[0] == 'typed_var' for arg in expression.args[:-1]), \
                f'Cannot parse aggregation expression: "{expression}", invalid product arguments!'

            # then combine linear functions resulting from param substitutions in sub-expressions
            param_maps = self._get_param_mappings(expression)
            lf = {CONSTANT: 1.}
            for p_map in param_maps:
                expr = self._convert_expression(expression.args[-1], {**param_map, **p_map})
                const_val = _get_const_val(expr)
                assert const_val is not None, \
                    f'Cannot parse aggregation: "{expression}", non-constant expression: "{expr}"!'
                lf = _scale_linear_function(lf, const_val)  # product of constant values
            return {CONSTANT: 0} if len(lf) == 0 else lf

        if d_type == 'forall':
            assert all(len(arg) == 2 and arg[0] == 'typed_var' for arg in expression.args[:-1]), \
                f'Cannot parse aggregation expression: "{expression}", invalid conjunction arguments!'

            # combine param substitutions in sub-expressions
            param_maps = self._get_param_mappings(expression)
            sub_exprs = []
            for p_map in param_maps:
                sub_exprs.append(self._convert_expression(expression.args[-1], {**param_map, **p_map}))

            # if all linear ops, just add everything (thresh >= len)
            if all(_is_linear_function(expr) for expr in sub_exprs):
                lf = {}
                for expr in sub_exprs:
                    lf = _combine_linear_functions(lf, expr)  # sum linear functions
                return {CONSTANT: False} if len(lf) == 0 else {'linear_and': lf}

            # otherwise build nested AND tree # TODO can do plane conjunction when supported in PsychSim
            lf = sub_exprs[0]
            for expr in sub_exprs[1:]:
                lf = {'logic_and': (lf, expr)}  # AND tree
            return lf

        if d_type == 'exists':
            assert all(len(arg) == 2 and arg[0] == 'typed_var' for arg in expression.args[:-1]), \
                f'Cannot parse aggregation expression: "{expression}", invalid disjunction arguments!'

            # combine param substitutions in sub-expressions
            param_maps = self._get_param_mappings(expression)
            sub_exprs = []
            for p_map in param_maps:
                sub_exprs.append(self._convert_expression(expression.args[-1], {**param_map, **p_map}))

            # if all linear ops, just add everything (linear OR)
            if all(_is_linear_function(expr) for expr in sub_exprs):
                lf = {}
                for expr in sub_exprs:
                    lf = _combine_linear_functions(lf, expr)  # sum linear functions
                return {CONSTANT: False} if len(lf) == 0 else {'linear_or': lf}

            # otherwise build nested OR tree # TODO can do plane disjunction when supported in PsychSim
            lf = sub_exprs[0]
            for expr in sub_exprs[1:]:
                lf = {'logic_or': (lf, expr)}  # AND tree
            return lf

        raise NotImplementedError(f'Cannot parse aggregation expression: "{expression}" of type "{d_type}"!')
