import logging
from typing import Dict, Tuple, Union
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
           all(type(v) in {float, int, bool} for v in expr.values())


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

        raise ValueError(f'Cannot parse expression, invalid relational operation between {lhs} and {rhs}!')

    def _get_nested_if(self, c: Dict, tb: Dict, fb: Dict, expression: Expression = None, agent: Agent = None) -> Dict:
        if 'pwl_and' in c and len(c) == 1:
            c = c['pwl_and']
            return {'if': (c, len([v for v in c.values() if v > 0]) - 0.5, 1),  # AND of features
                    True: tb,
                    False: fb}

        if 'logic_and' in c and len(c) == 1:
            lhs, rhs = c['logic_and']  # composes nested AND tree
            return {'if': lhs,
                    True: self._get_nested_if(rhs, tb, fb, expression, agent),
                    False: fb}

        if 'pwl_or' in c and len(c) == 1:
            c = c['pwl_or']
            return {'if': (c, 0.5, 1),  # OR of features (only 1 needs to be true)
                    True: tb,
                    False: fb}

        if 'logic_or' in c and len(c) == 1:
            lhs, rhs = c['logic_or']  # composes nested OR tree
            return {'if': lhs,
                    True: tb,
                    False: self._get_nested_if(rhs, tb, fb, expression, agent)}

        if 'not' in c and len(c) == 1:
            return self._get_nested_if(c['not'], fb, tb, expression, agent)  # if NOT, just flip branches

        if 'equiv' in c:
            lhs, rhs = c['equiv']
            lhs = _combine_linear_functions(lhs, _negate_linear_function(rhs))
            return {'if': (lhs, 0, 0),  # takes equality of pwl comb in vectors (difference==0)
                    True: tb,
                    False: fb}

        if 'imply' in c:
            # if IMPLICATION, false only if left is true and right is false
            lhs, rhs = c['imply']
            return {'if': (lhs, 0.5, 1),  # if left is true (> 0.5)
                    True: {'if': (rhs, 0.5, 1),  # if right is true (> 0.5)
                           True: tb,
                           False: fb},
                    False: tb}

        if 'eq' in c:
            lhs, rhs = c['eq']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, 0),  # takes equality of pwl comb in vectors (difference==0)
                    True: tb,
                    False: fb}

        if 'neq' in c:
            lhs, rhs = c['neq']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, 0),  # takes equality of pwl comb in vectors (difference==0)
                    True: fb,  # then switch branches
                    False: tb}

        if 'gt' in c:
            lhs, rhs = c['gt']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, 1),  # takes diff of pwl comb in vectors (difference>0)
                    True: tb,
                    False: fb}

        if 'lt' in c:
            lhs, rhs = c['lt']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, -1),  # takes diff of pwl comb in vectors (difference<0)
                    True: tb,
                    False: fb}

        if 'geq' in c:
            lhs, rhs = c['geq']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, -1),  # takes diff of pwl comb in vectors (difference<0)
                    True: fb,  # then switch branches
                    False: tb}

        if 'leq' in c:
            lhs, rhs = c['leq']
            op, thresh = self._get_relational_plane_thresh(lhs, rhs)
            return {'if': (op, thresh, 1),  # takes diff of pwl comb in vectors (difference>0)
                    True: fb,  # then switch branches
                    False: tb}

        if 'action' in c and len(c) == 1 and agent is not None:
            return {'if': ({actionKey(agent.name): 1.}, c['action'], 0),  # conditional on specific agent's action
                    True: tb,
                    False: fb}

        if _is_linear_function(c):
            return {'if': (c, 0, 1),  # default: assumes linear combination of all features in vector has to be > 0
                    True: tb,
                    False: fb}

        raise ValueError(f'Could not parse RDDL expression "{expression}", invalid nested PWL control in "{c}"!')

    def _convert_expression(self, expression: Expression, agent: Agent) -> Dict:

        # process leaf node, try to get feature name or constant value
        e_type = expression.etype[0]
        if e_type == 'constant':
            try:
                return {CONSTANT: expression.args}
            except ValueError as e:
                logging.info(f'Could not convert value "{expression.args}" to float in RDDL expression "{expression}"!')
                raise e

        if e_type == 'penum':
            # just check if enumerated type is known
            val = expression.args.replace('@', '')
            if self._is_enum_type(val):
                return {CONSTANT: val}
            raise ValueError(f'Could not find enumerated type from RDDL expression "{expression}"!')

        if e_type == 'pvar':
            name = expression.args[0]
            if self._is_feature(name):  # feature
                return {self._get_feature(name): 1.}

            if self._is_action(name, agent):
                return {'action': self._get_action(name, agent)}  # identify this as the agent's action

            if self._is_constant(name):  # named constant
                try:
                    value = self._get_constant_value(name)
                    return {CONSTANT: float(value)}
                except ValueError as e:
                    logging.info(f'Could not convert value "{value}" to float in RDDL expression "{expression}"!')
                    raise e

            raise ValueError(f'Could not find feature, action or constant from RDDL expression "{expression}"!')

        if e_type == 'arithmetic':
            return self._convert_arithmetic_expr(expression, agent)

        if e_type == 'boolean':
            return self._convert_boolean_expr(expression, agent)

        if e_type == 'relational':
            return self._convert_relational(expression, agent)

        if e_type == 'control':
            return self._convert_control_expr(expression, agent)

        if e_type == 'randomvar':
            return self._convert_distribution_expr(expression, agent)

        # not yet implemented
        raise NotImplementedError(f'Cannot parse expression: "{expression}" of type "{e_type}"!')

    def _convert_arithmetic_expr(self, expression: Expression, agent: Agent) -> Dict:

        lhs = self._convert_expression(expression.args[0], agent)
        rhs = self._convert_expression(expression.args[1], agent) if len(expression.args) > 1 else {}
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
                return {k: lhs_const * v for k, v in rhs.items()}  # multiply right-hand side by const
            if isinstance(rhs_const, float) and _is_linear_function(lhs):
                return {k: rhs_const * v for k, v in lhs.items()}  # multiply left-hand side by const
            raise ValueError(f'Non-PWL operation not supported: "{expression}"!')

        elif a_type == '/':
            # if division, only works if right or both sides are constants
            if all_consts:
                return {CONSTANT: lhs_const / rhs_const}  # reduce
            if isinstance(rhs_const, float) and _is_linear_function(lhs):
                return {k: v / rhs_const for k, v in lhs.items()}  # divide left-hand side by const
            raise ValueError(f'Non-PWL operation not supported: "{expression}"!')

        raise NotImplementedError(f'Cannot parse arithmetic expression: "{expression}" of type "{a_type}"!')

    def _convert_boolean_expr(self, expression: Expression, agent: Agent) -> Dict:

        lhs = self._convert_expression(expression.args[0], agent)
        rhs = self._convert_expression(expression.args[1], agent) if len(expression.args) > 1 else {}
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
            if 'pwl_and' in lhs and len(lhs) == 1:
                lhs = lhs['pwl_and']
            orig_rhs = rhs
            if 'pwl_and' in rhs and len(rhs) == 1:
                rhs = rhs['pwl_and']
            if _is_linear_function(lhs) and _is_linear_function(rhs):
                # if both linear ops, just add everything from both sides (thresh >= len)
                return {'pwl_and': _combine_linear_functions(lhs, rhs)}
            return {'logic_and': (orig_lhs, orig_rhs)}  # defer for later processing

        if b_type == '|':
            # if OR, one side has to be True
            if all_consts:
                return {CONSTANT: bool(rhs_const) or bool(lhs_const)}
            if isinstance(rhs_const, float):
                return {CONSTANT: True} if bool(rhs_const) else lhs
            if isinstance(lhs_const, float):
                return {CONSTANT: True} if bool(lhs_const) else rhs
            orig_lhs = lhs
            if 'pwl_or' in lhs and len(lhs) == 1:
                lhs = lhs['pwl_or']
            orig_rhs = rhs
            if 'pwl_or' in rhs and len(rhs) == 1:
                rhs = rhs['pwl_or']
            if _is_linear_function(lhs) and _is_linear_function(rhs):
                # if both vectors, just add everything from both sides (thresh > 0)
                return {'pwl_or': _combine_linear_functions(lhs, rhs)}
            return {'logic_or': (orig_lhs, orig_rhs)}  # defer for later processing

        if b_type == '~':
            # NOT
            if isinstance(lhs_const, float):
                return {CONSTANT: False if bool(lhs_const) else True}  # gets constant's truth value
            if 'not' in lhs and len(lhs) == 1:
                return lhs['not']  # double negation
            return {'not': lhs}  # defer for later processing

        if b_type == '<=>':
            # if EQUIV, sides have to be of equal boolean value
            if all_consts:
                return {CONSTANT: bool(rhs_const) == bool(lhs_const)}  # equal booleans
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}

            # left and right have to be single variables
            assert len(lhs) == 1 and _is_linear_function(lhs) and \
                   len(rhs) == 1 and _is_linear_function(rhs), \
                f'Could not parse boolean expression "{expression}", invalid PWL equivalence composition!'
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
                if _is_linear_function(lhs):
                    return {'not': lhs}  # right is false, negate left
                return {'imply': lhs}  # defer for later processing

            # left and right have to be single variables
            assert len(lhs) == 1 and _is_linear_function(lhs) and \
                   len(rhs) == 1 and _is_linear_function(rhs), \
                f'Could not parse boolean expression "{expression}", invalid PWL equivalence composition!'
            return {'imply': (lhs, rhs)}  # defer for later processing

        raise NotImplementedError(f'Cannot parse boolean expression: "{expression}" of type "{b_type}"!')

    def _convert_relational(self, expression: Expression, agent: Agent) -> Dict:

        lhs = self._convert_expression(expression.args[0], agent)
        rhs = self._convert_expression(expression.args[1], agent) if len(expression.args) > 1 else {}
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

            # left and right have to be pwl
            assert _is_linear_function(lhs) and _is_linear_function(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'eq': (lhs, rhs)}  # defer for later processing

        if b_type == '~=':
            # if NOT_EQUAL, sides have to be of different value
            if all_consts:
                return {CONSTANT: lhs_const != rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}

            # left and right have to be pwl
            assert _is_linear_function(lhs) and _is_linear_function(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'neq': (lhs, rhs)}  # defer for later processing

        if b_type == '>':
            # if GREATER_THAN, left has to be of higher value than right
            if all_consts:
                return {CONSTANT: lhs_const > rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}

            # left and right have to be pwl
            assert _is_linear_function(lhs) and _is_linear_function(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'gt': (lhs, rhs)}  # defer for later processing

        if b_type == '<':
            # if LESS_THAN, left has to be of lower value than right
            if all_consts:
                return {CONSTANT: lhs_const < rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}

            # left and right have to be pwl
            assert _is_linear_function(lhs) and _is_linear_function(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'lt': (lhs, rhs)}  # defer for later processing

        if b_type == '>=':
            # if GREATER_OR_EQUAL, left has to be of equal or higher value than right
            if all_consts:
                return {CONSTANT: lhs_const >= rhs_const}
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}

            # left and right have to be pwl
            assert _is_linear_function(lhs) and _is_linear_function(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'geq': (lhs, rhs)}  # defer for later processing

        if b_type == '<=':
            # if LESS_OR_EQUAL, left has to be of equal or lower value than right
            if all_consts:
                return {CONSTANT: lhs_const <= rhs_const}
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}

            # left and right have to be pwl
            assert _is_linear_function(lhs) and _is_linear_function(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'leq': (lhs, rhs)}  # defer for later processing

        raise NotImplementedError(f'Cannot parse relational expression: "{expression}" of type "{b_type}"!')

    def _convert_control_expr(self, expression: Expression, agent: Agent) -> Dict:
        c_type = expression.etype[1]
        if c_type == 'if':
            # get condition and branches
            cond = self._convert_expression(expression.args[0], agent)
            true_branch = self._convert_expression(expression.args[1], agent)
            false_branch = self._convert_expression(expression.args[2], agent)
            return self._get_nested_if(cond, true_branch, false_branch, expression, agent)

        if c_type == 'switch':
            assert len(expression.args) > 1, f'Cannot parse switch expression: "{expression}", no cases provided!'

            # get expression for terminal condition, has to be PWL
            cond = self._convert_expression(expression.args[0], agent)
            assert _is_linear_function(
                cond), f'Cannot parse switch expression: "{expression}", switch condition is not PWL!'

            # get expressions for each of the branches
            case_values = []
            case_branches = []
            for arg in expression.args[1:]:
                case_type = arg[0]
                if case_type == 'case':
                    val = self._convert_expression(arg[1][0], agent)
                    branch = self._convert_expression(arg[1][1], agent)
                elif case_type == 'default':
                    assert 'default' not in case_values, f'Cannot parse switch expression: "{expression}", ' \
                                                         f'default branch defined more than once!'
                    val = 'default'
                    branch = self._convert_expression(arg[1], agent)
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

    def _convert_distribution_expr(self, expression: Expression, agent: Agent) -> Dict:
        d_type = expression.etype[1]
        if d_type == 'Bernoulli':
            arg = self._convert_expression(expression.args[0], agent)
            assert _get_const_val(arg) is not None, \
                f'Cannot parse stochastic expression: "{expression}", non-constant probability: "{arg}"!'
            p = arg[CONSTANT]
            return {'distribution': [(1, p), (0, 1 - p)]}

        if d_type == 'KronDelta':
            # return the argument itself, although result should be int? From the docs:
            # "places all probability mass on its discrete argument v, discrete sample is thus deterministic"
            return self._convert_expression(expression.args[0], agent)

        if d_type == 'DiracDelta':
            # return the argument itself. From the docs:
            # "places all probability mass on its continuous argument v, continuous sample is thus deterministic"
            return self._convert_expression(expression.args[0], agent)

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
                v = self._convert_expression(arg[1][1], agent)
                assert _get_const_val(v) is not None, \
                    f'Cannot parse stochastic expression: "{expression}", non-constant probability: "{v}"!'
                dist.append((k, v[CONSTANT]))
            return {'distribution': dist}

        raise NotImplementedError(f'Cannot parse stochastic expression: "{expression}" of type "{d_type}"!')
