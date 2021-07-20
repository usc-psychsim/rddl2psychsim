import logging
from typing import Dict, Tuple, Union
from pyrddl.expr import Expression
from psychsim.action import ActionSet
from psychsim.agent import Agent
from psychsim.pwl import KeyedTree, makeFuture, makeTree, CONSTANT, KeyedPlane, KeyedVector
from rddl2psychsim.conversion.dynamics import _DynamicsConverter
from rddl2psychsim.conversion.expression import _get_const_val, _is_linear_function
from rddl2psychsim.rddl import expression_to_rddl

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

_ASSERTION_KEY = '__ASSERT__'


class _ConstraintsConverter(_DynamicsConverter):

    def __init__(self, const_as_assert=True, **kwargs):
        super().__init__(**kwargs)
        self._const_as_assert = const_as_assert
        self._constraint_trees: Dict[KeyedTree, Expression] = {}

    def verify_constraints(self) -> None:
        """
        Verifies the constraints stored from the RDDL definition against the current state of the world.
        For each unsatisfied constraint, an `AssertionError` is thrown if `const_as_assert` parameter is `True`,
        otherwise a message is sent via `logging.info`.
        """
        # verifies if all constraints are satisfied by verifying the trees against current state
        for tree, expr in self._constraint_trees.items():
            state = self.world.state.copySubset()
            state *= tree
            val = state.marginal(makeFuture(_ASSERTION_KEY)).expectation()  # expected / mean value
            # value has to be > 0.5, which is truth value in PsychSim (see psychsim.world.World.float2value)
            if val <= 0.5:
                err_msg = f'State or action constraint "{expression_to_rddl(expr)}" ' \
                          f'not satisfied given current world state:\n{state}'
                if self._const_as_assert:
                    raise AssertionError(err_msg)
                logging.info(err_msg)

    def _convert_state_action_constraints(self):
        logging.info('__________________________________________________')
        for constraint in self.model.domain.constraints + self.model.domain.preconds:

            # check param action legality constraint in the form "forall_p action(p) => constraint" before conversion
            if constraint.etype[0] == 'aggregation' and constraint.etype[1] == 'forall' and \
                    constraint.args[-1].etype[0] == 'boolean' and constraint.args[-1].etype[1] == '=>':
                # combine param substitutions in sub-expression and tries to set legality for each instantiation
                for param_map in self._get_param_mappings(constraint):
                    self._convert_state_action_constraint(constraint.args[-1], param_map)
            else:
                # otherwise process normal constraint expression
                self._convert_state_action_constraint(constraint)

        logging.info(f'Total {len(self._constraint_trees)} dynamic state constraints created')

    def _convert_state_action_constraint(self, constraint: Expression, param_map: Dict[str, str] = None):
        expr = self._convert_expression(constraint, param_map)

        # if it's a constant-value constraint, we can assert it right now, at conversion time
        const_val = _get_const_val(expr)
        if const_val is not None:
            if not bool(const_val):
                err_msg = f'Constant state or action constraint "{constraint}" not satisfied!'
                if self._const_as_assert:
                    raise AssertionError(err_msg)
                logging.info(err_msg)
            logging.info(f'State or action constraint "{constraint}" is always satisfied')
            return

        # check for non-param action legality constraint in the form "action => constraint"
        legality_const = self._get_action_legality(expr)
        if legality_const is True:
            logging.info(f'Action constraint "{constraint}" is always satisfied')
            return
        if legality_const is not None:
            agent, action, legal_tree = legality_const
            agent.legal[action] = legal_tree.desymbolize(self.world.symbols)
            logging.info(f'Set legality constraint for action "{action}" to:\n{legal_tree}')
            return

        # otherwise store dynamics tree for a (external) boolean variable, for later online assertion
        tree = self._get_dynamics_tree(_ASSERTION_KEY, expr)
        legal_tree = tree.desymbolize(self.world.symbols)
        self._constraint_trees[legal_tree] = constraint
        logging.info(f'Added dynamic state constraint:\n{tree}')

    def _get_action_legality(self, expr: Dict) -> Union[Tuple[Agent, ActionSet, KeyedTree], None, bool]:
        # check for action legality constraint in the form "action => constraint"
        if 'imply' in expr and 'action' in expr['imply'][0]:
            agent = expr['imply'][0]['action'][0]
            action = expr['imply'][0]['action'][1]
            # get condition expression as 'if' legality tree
            legal_tree = self._get_legality_tree(expr['imply'][1])
            return agent, action, legal_tree

        if 'action' in expr or _get_const_val(expr, bool):
            # if constraint is true or only depends on action, then no need to set legality (always legal)
            return True

        if 'not' in expr and 'action' in expr['not']:
            # if "not action", probably rhs of implication was False, so action always illegal
            agent = expr['not']['action'][0]
            action = expr['not']['action'][1]
            return agent, action, makeTree(False)

        return None  # could not find action legality constraint in the expression

    def _get_legality_tree(self, expr: Dict) -> Dict or bool:

        def _leaf_func(leaf_expr: Dict) -> Dict:
            # get truth value of linear function (sum > 0.5 in PsychSim, see psychsim.world.World.float2value)
            if _is_linear_function(leaf_expr) or self._is_constant_expr(leaf_expr):
                const_val = _get_const_val(leaf_expr, bool)  # if constant just return boolean value
                return const_val if const_val is not None else \
                    {'if': KeyedPlane(KeyedVector(leaf_expr), 0.5, 1),
                     True: True,
                     False: False}
            raise NotImplementedError(f'Could not parse RDDL expression, got invalid legality subtree: "{leaf_expr}"!')

        # return a legality decision tree from the expression
        return self._get_decision_tree(expr, _leaf_func)
