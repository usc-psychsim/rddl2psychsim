import logging
from typing import Dict
from pyrddl.expr import Expression
from psychsim.pwl import KeyedTree, makeFuture, makeTree, CONSTANT
from rddl2psychsim.conversion.dynamics import _DynamicsConverter
from rddl2psychsim.conversion.expression import _get_const_val

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

_ASSERTION_KEY = '__ASSERT__'


class _ConstraintsConverter(_DynamicsConverter):

    def __init__(self, const_as_assert=True):
        super().__init__()
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
                err_msg = f'State or action constraint "{expr}" not satisfied, returned False!'
                if self._const_as_assert:
                    raise AssertionError(err_msg)
                logging.info(err_msg)

    def _convert_state_action_constraints(self):
        for constraint in self.model.domain.constraints:
            expr = self._convert_expression(constraint)

            # if it's a constant-value constraint, we can assert it right at conversion time
            const_val = _get_const_val(expr)
            if const_val is not None:
                if not bool(const_val):
                    err_msg = f'Constant state or action constraint "{constraint}" not satisfied, returned False!'
                    if self._const_as_assert:
                        raise AssertionError(err_msg)
                    logging.info(err_msg)
                continue

            # otherwise store dynamics tree for a boolean variable, for later online assertion
            tree = makeTree(self._get_dynamics_tree(
                _ASSERTION_KEY, self._get_pwl_tree(expr, {CONSTANT: True}, {CONSTANT: False})))
            tree = tree.desymbolize(self.world.symbols)
            self._constraint_trees[tree] = constraint
