from pyrddl.expr import Expression
from pyrddl.parser import RDDLParser
from pyrddl.rddl import RDDL

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def parse_rddl_file(rddl_file: str, verbose: bool) -> RDDL:
    # read RDDL file
    with open(rddl_file, 'r') as file:
        rddl = file.read()

    return parse_rddl(rddl, verbose)


def parse_rddl(rddl: str, verbose: bool) -> RDDL:
    # parse RDDL
    rddl_parser = RDDLParser(verbose=verbose)
    # rddl_parser.debugging = verbose
    rddl_parser.build()
    return rddl_parser.parse(rddl)


def _add_parens(expr: str) -> str:
    return expr if expr.startswith('(') and expr.endswith(')') else f'({expr})'


def expression_to_rddl(expr: Expression, level: int = 0, indent_width: int = 4) -> str:
    """
    Transforms a pyrddl Expression into a valid RDDL expression string representation.
    :param Expression expr: the pyrddl expression that we wanto to convert.
    :param int level: the text indentation level at which to write the expression.
    :param int indent_width: the width of one level of text indentation, i.e., number of white space characters.
    :rtype: str
    :return: a string corresponding to the RDDL expression.
    """
    indent = ' ' * level * indent_width
    if isinstance(expr, str):
        return indent + expr

    if isinstance(expr, tuple):
        if expr[0] == 'enum_type':
            return f'{indent}{str(expr[1])}'
        return f'{indent}{str(expr)}'

    e_type, e_name = expr.etype
    args = expr.args

    if e_type == 'pvar':
        params = f'({", ".join(args[1])})' if len(args) > 1 and args[1] is not None else ''
        return indent + e_name + params

    if e_type == 'penum':
        return indent + str(args)

    if e_type == 'param':
        return indent + str(args)

    if e_type == 'constant':
        return indent + str(args).lower()

    if e_type in ['arithmetic', 'boolean', 'relational']:
        if len(args) == 1:
            return f'{indent}{e_name}{expression_to_rddl(args[0], 0)}'
        return f'{indent}({expression_to_rddl(args[0], 0)} {e_name} {expression_to_rddl(args[1], 0)})'

    if e_type == 'aggregation':
        p_types = ', '.join(f'{p_type[1][0]}: {p_type[1][1]}' for p_type in args[:-1])
        return f'{e_name}_{{{p_types}}}[{expression_to_rddl(args[-1], 0)}]'

    if e_name == 'if':
        if_cond = _add_parens(expression_to_rddl(args[0], 0))
        return f'{indent}if {if_cond} then' \
               f'\n{expression_to_rddl(args[1], level + 1)}' \
               f'\n{indent}else\n{expression_to_rddl(args[2], level + 1)}'

    if e_name == 'switch':
        switch_cond = _add_parens(expression_to_rddl(args[0], 0))
        case_conds = ''
        inner_indent = ' ' * (level + 1) * indent_width
        for case_type, val_expr in args[1:]:
            case_conds += f'\n{inner_indent}{case_type}'
            if case_type == 'default':
                case_conds += f': {expression_to_rddl(val_expr, 0)},'
            else:
                case_conds += f' {expression_to_rddl(val_expr[0], 0)}' \
                              f': {expression_to_rddl(val_expr[1], 0)},'
        return f'{indent}switch {switch_cond}{{{case_conds[:-1]}}}'

    if e_type == 'randomvar':

        if e_name == 'Discrete':
            dist_expr = _add_parens(expression_to_rddl(args[0], 0))
            inner_indent = ' ' * (level + 1) * indent_width
            probs = ''
            for var_prob in args[1:]:
                var_expr, prob_expr = var_prob[1]
                probs += f'\n{inner_indent}{expression_to_rddl(var_expr, 0)}: {expression_to_rddl(prob_expr, 0)},'
            return f'{indent}{e_name} {dist_expr}{{{probs[:-1]}}}'

        dist_expr = _add_parens(', '.join(expression_to_rddl(arg, 0) for arg in args))
        return f'{indent}{e_name}{dist_expr}'

    return f'{indent}{e_name}({", ".join(expression_to_rddl(arg, 0) for arg in args)})'
