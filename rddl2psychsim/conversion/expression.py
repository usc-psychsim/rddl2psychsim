import logging
from typing import Dict
from pyrddl.expr import Expression
from psychsim.agent import Agent
from psychsim.pwl import CONSTANT, actionKey
from rddl2psychsim.conversion import _ConverterBase

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def is_pwl_op(o) -> bool:
    return isinstance(o, dict) and \
           all(isinstance(k, str) for k in o.keys()) and \
           all(type(v) in {float, int, bool, str} for v in o.values())


def nested_if(c, tb, fb, expression: Expression = None, agent: Agent = None) -> Dict:
    if 'pwl_and' in c and len(c) == 1:
        c = c['pwl_and']
        return {'if': (c, len([v for v in c.values() if v > 0]) - 0.5, 1),  # AND of features
                True: tb,
                False: fb}

    if 'logic_and' in c and len(c) == 1:
        lhs, rhs = c['logic_and']  # composes nested AND tree
        return {'if': lhs,
                True: nested_if(rhs, tb, fb, expression, agent),
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
                False: nested_if(rhs, tb, fb, expression, agent)}

    if 'not' in c and len(c) == 1:
        return nested_if(c['not'], fb, tb, expression, agent)  # if NOT, just flip branches

    if 'equiv' in c:
        lhs, rhs = c['equiv']
        lhs = dict(lhs)
        _update_weights(lhs, _negate_weights(rhs))
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
        lhs = dict(lhs)
        _update_weights(lhs, _negate_weights(rhs))
        return {'if': (lhs, 0, 0),  # takes equality of pwl comb in vectors (difference==0)
                True: tb,
                False: fb}

    if 'neq' in c:
        lhs, rhs = c['neq']
        lhs = dict(lhs)
        _update_weights(lhs, _negate_weights(rhs))
        return {'if': (lhs, 0, 0),  # takes equality of pwl comb in vectors (difference==0)
                True: fb,  # then switch branches
                False: tb}

    if 'gt' in c:
        lhs, rhs = c['gt']
        lhs = dict(lhs)
        _update_weights(lhs, _negate_weights(rhs))
        return {'if': (lhs, 0, 1),  # takes diff of pwl comb in vectors (difference>0)
                True: tb,
                False: fb}

    if 'lt' in c:
        lhs, rhs = c['lt']
        lhs = dict(lhs)
        _update_weights(lhs, _negate_weights(rhs))
        return {'if': (lhs, 0, -1),  # takes diff of pwl comb in vectors (difference<0)
                True: tb,
                False: fb}

    if 'geq' in c:
        lhs, rhs = c['geq']
        lhs = dict(lhs)
        _update_weights(lhs, _negate_weights(rhs))
        return {'if': (lhs, 0, -1),  # takes diff of pwl comb in vectors (difference<0)
                True: fb,  # then switch branches
                False: tb}

    if 'leq' in c:
        lhs, rhs = c['leq']
        lhs = dict(lhs)
        _update_weights(lhs, _negate_weights(rhs))
        return {'if': (lhs, 0, 1),  # takes diff of pwl comb in vectors (difference>0)
                True: fb,  # then switch branches
                False: tb}

    if 'action' in c and len(c) == 1 and agent is not None:
        return {'if': ({actionKey(agent.name): 1.}, c['action'], 0),  # conditional on specific agent's action
                True: tb,
                False: fb}

    if is_pwl_op(c):
        return {'if': (c, 0, 1),  # assumes linear combination of all features in vector has to be > 0
                True: tb,
                False: fb}

    raise ValueError(f'Could not parse RDDL expression "{expression}", invalid nested PWL control in "{c}"!')


def _get_const_val(s) -> float or None:
    return s if isinstance(s, float) else float(s[CONSTANT]) if len(s) == 1 and CONSTANT in s else None


def _update_weights(old_weights, new_weights):
    assert is_pwl_op(new_weights), f'Could not parse RDDL expression, invalid PWL operation in "{new_weights}"!'
    for k, v in new_weights.items():
        old_weights[k] = old_weights[k] + v if k in old_weights else v  # add weight if key already in dict
        if old_weights[k] == 0:
            del old_weights[k]  # remove if weight is 0


def _negate_weights(weights):
    assert is_pwl_op(weights), f'Could not parse RDDL expression, invalid PWL operation in "{weights}"!'
    return {k: -v for k, v in weights.items()}  # just negate the weight


class _ExpressionConverter(_ConverterBase):

    def __init__(self):
        super().__init__()

    def _get_expression_dict(self, expression: Expression, agent: Agent) -> Dict:

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
            name = expression.args
            if self._is_enum_type(name):
                return {name: 1.}
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

        lhs = self._get_expression_dict(expression.args[0], agent)
        rhs = self._get_expression_dict(expression.args[1], agent) if len(expression.args) > 1 else {}
        lhs_const = _get_const_val(lhs)
        rhs_const = _get_const_val(rhs) if rhs is not None else None
        all_consts = isinstance(lhs_const, float) and isinstance(rhs_const, float)
        weights = {}
        a_type = expression.etype[1]

        if a_type == '+':
            if all_consts:
                return {CONSTANT: lhs_const + rhs_const}  # reduce
            # if addition, just add everything from both sides
            _update_weights(weights, lhs)
            _update_weights(weights, rhs)
            return weights

        if a_type == '-':
            if all_consts:
                return {CONSTANT: lhs_const - rhs_const}  # reduce
            # if subtraction, get left-hand side
            if len(rhs) == 0:
                rhs = lhs  # just switch if we only have one argument
                lhs = {}
            _update_weights(weights, lhs)
            _update_weights(weights, {k: -v for k, v in rhs.items()})  # then multiply right-hand side by -1
            return weights

        if a_type == '*':
            # if multiplication, only works if one or both sides are constants
            if all_consts:
                return {CONSTANT: lhs_const * rhs_const}  # reduce
            if isinstance(lhs_const, float):
                return {k: lhs_const * v for k, v in rhs.items()}  # multiply right-hand side by const
            if isinstance(rhs_const, float):
                return {k: rhs_const * v for k, v in lhs.items()}  # multiply left-hand side by const
            raise ValueError(f'Non-PWL operation not supported: "{expression}"!')

        elif a_type == '/':
            # if division, only works if right or both sides are constants
            if all_consts:
                return {CONSTANT: lhs_const / rhs_const}  # reduce
            if isinstance(rhs_const, float):
                return {k: v / rhs_const for k, v in lhs.items()}  # divide left-hand side by const
            raise ValueError(f'Non-PWL operation not supported: "{expression}"!')

        raise NotImplementedError(f'Cannot parse arithmetic expression: "{expression}" of type "{a_type}"!')

    def _convert_boolean_expr(self, expression: Expression, agent: Agent) -> Dict:

        lhs = self._get_expression_dict(expression.args[0], agent)
        rhs = self._get_expression_dict(expression.args[1], agent) if len(expression.args) > 1 else {}
        lhs_const = _get_const_val(lhs)
        rhs_const = _get_const_val(rhs)
        all_consts = isinstance(lhs_const, float) and isinstance(rhs_const, float)

        weights = {}
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
            if is_pwl_op(lhs) and is_pwl_op(rhs):
                _update_weights(weights, lhs)  # if both vectors, just add everything from both sides (thresh >= len)
                _update_weights(weights, rhs)
                return {'pwl_and': weights}
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
            if is_pwl_op(lhs) and is_pwl_op(rhs):
                _update_weights(weights, lhs)  # if both vectors, just add everything from both sides (thresh > 0)
                _update_weights(weights, rhs)
                return {'pwl_or': weights}
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
            assert len(lhs) == 1 and is_pwl_op(lhs) and \
                   len(rhs) == 1 and is_pwl_op(rhs), \
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
                if is_pwl_op(lhs):
                    return {'not': lhs}  # right is false, negate left
                return {'imply': lhs}  # defer for later processing

            # left and right have to be single variables
            assert len(lhs) == 1 and is_pwl_op(lhs) and \
                   len(rhs) == 1 and is_pwl_op(rhs), \
                f'Could not parse boolean expression "{expression}", invalid PWL equivalence composition!'
            return {'imply': (lhs, rhs)}  # defer for later processing

        raise NotImplementedError(f'Cannot parse boolean expression: "{expression}" of type "{b_type}"!')

    def _convert_relational(self, expression: Expression, agent: Agent) -> Dict:

        lhs = self._get_expression_dict(expression.args[0], agent)
        rhs = self._get_expression_dict(expression.args[1], agent) if len(expression.args) > 1 else {}
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
            assert is_pwl_op(lhs) and is_pwl_op(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'eq': (lhs, rhs)}  # defer for later processing

        if b_type == '~=':
            # if NOT_EQUAL, sides have to be of different value
            if all_consts:
                return {CONSTANT: lhs_const != rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}

            # left and right have to be pwl
            assert is_pwl_op(lhs) and is_pwl_op(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'neq': (lhs, rhs)}  # defer for later processing

        if b_type == '>':
            # if GREATER_THAN, left has to be of higher value than right
            if all_consts:
                return {CONSTANT: lhs_const > rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}

            # left and right have to be pwl
            assert is_pwl_op(lhs) and is_pwl_op(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'gt': (lhs, rhs)}  # defer for later processing

        if b_type == '<':
            # if LESS_THAN, left has to be of lower value than right
            if all_consts:
                return {CONSTANT: lhs_const < rhs_const}
            if rhs == lhs:  # equal dicts, so not different
                return {CONSTANT: False}

            # left and right have to be pwl
            assert is_pwl_op(lhs) and is_pwl_op(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'lt': (lhs, rhs)}  # defer for later processing

        if b_type == '>=':
            # if GREATER_OR_EQUAL, left has to be of equal or higher value than right
            if all_consts:
                return {CONSTANT: lhs_const >= rhs_const}
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}

            # left and right have to be pwl
            assert is_pwl_op(lhs) and is_pwl_op(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'geq': (lhs, rhs)}  # defer for later processing

        if b_type == '<=':
            # if LESS_OR_EQUAL, left has to be of equal or lower value than right
            if all_consts:
                return {CONSTANT: lhs_const <= rhs_const}
            if rhs == lhs:  # equal dicts
                return {CONSTANT: True}

            # left and right have to be pwl
            assert is_pwl_op(lhs) and is_pwl_op(rhs), \
                f'Could not parse relational expression "{expression}", invalid PWL equivalence composition!'
            return {'leq': (lhs, rhs)}  # defer for later processing

        raise NotImplementedError(f'Cannot parse relational expression: "{expression}" of type "{b_type}"!')

    def _convert_control_expr(self, expression: Expression, agent: Agent) -> Dict:
        c_type = expression.etype[1]
        if c_type == 'if':
            # get condition and branches
            cond = self._get_expression_dict(expression.args[0], agent)
            true_branch = self._get_expression_dict(expression.args[1], agent)
            false_branch = self._get_expression_dict(expression.args[2], agent)
            return nested_if(cond, true_branch, false_branch, expression, agent)
        else:
            raise NotImplementedError(f'Cannot parse control expression: "{expression}" of type "{c_type}"!')

    def _convert_distribution_expr(self, expression: Expression, agent: Agent) -> Dict:
        d_type = expression.etype[1]
        if d_type == 'Bernoulli':
            arg = self._get_expression_dict(expression.args[0], agent)
            assert len(arg) == 1 and CONSTANT in arg, \
                f'Cannot parse stochastic expression: "{expression}", non-constant probability: "{arg}"!'
            p = arg[CONSTANT]
            return {'distribution': [(1., p), (0., 1 - p)]}

        if d_type == 'KronDelta':
            # return
            arg = self._get_expression_dict(expression.args[0], agent)
            return nested_if(arg, {CONSTANT: 1.}, {CONSTANT: 0.}, expression, agent)

        else:
            raise NotImplementedError(f'Cannot parse stochastic expression: "{expression}" of type "{d_type}"!')
