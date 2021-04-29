import logging
from typing import Dict, Tuple, Union
from pyrddl.expr import Expression
from psychsim.action import ActionSet
from psychsim.agent import Agent
from psychsim.pwl import KeyedTree, makeFuture, makeTree, CONSTANT, KeyedPlane, KeyedVector
from rddl2psychsim.conversion.dynamics import _DynamicsConverter
from rddl2psychsim.conversion.expression import _get_const_val
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
        for constraint in self.model.domain.constraints:

            # check param action legality constraint in the form "forall_p action(p) => constraint" before conversion
            if constraint.etype[0] == 'aggregation' and constraint.etype[1] == 'forall' and \
                    constraint.args[-1].etype[0] == 'boolean' and constraint.args[-1].etype[1] == '=>':

                # combine param substitutions in sub-expression and tries to set legality for each instantiation
                param_maps = self._get_param_mappings(constraint)
                legality_consts = [self._get_action_legality(self._convert_expression(constraint.args[-1], p_map))
                                   for p_map in param_maps]
                if None not in legality_consts:  # we should get a legality tree for all agents, otherwise invalid
                    for agent, action, legal_tree in legality_consts:
                        agent.legal[action] = legal_tree
                        logging.info(f'Set legality constraint for action "{action}" to:\n{legal_tree}')
                    continue  # safely consumed action legality constraints

            # otherwise process constraint expression
            expr = self._convert_expression(constraint)

            # if it's a constant-value constraint, we can assert it right now, at conversion time
            const_val = _get_const_val(expr)
            if const_val is not None:
                if not bool(const_val):
                    err_msg = f'Constant state or action constraint "{constraint}" not satisfied!'
                    if self._const_as_assert:
                        raise AssertionError(err_msg)
                    logging.info(err_msg)
                continue

            # check for non-param action legality constraint in the form "action => constraint"
            legality_const = self._get_action_legality(expr)
            if legality_const is not None:
                agent, action, legal_tree = legality_const
                agent.legal[action] = legal_tree
                logging.info(f'Set legality constraint for action "{action}" to:\n{legal_tree}')
                continue

            # otherwise store dynamics tree for a (external) boolean variable, for later online assertion
            tree = makeTree(self._get_dynamics_tree(
                _ASSERTION_KEY, self._get_pwl_tree(expr, {CONSTANT: True}, {CONSTANT: False})))
            tree = tree.desymbolize(self.world.symbols)
            self._constraint_trees[tree] = constraint
            logging.info(f'Added dynamic state constraint:\n{tree}')

        logging.info(f'Total {len(self._constraint_trees)} constraints created')

    def _get_action_legality(self, expr: Dict) -> Union[Tuple[Agent, ActionSet, KeyedTree], None]:
        # check for action legality constraint in the form "action => constraint"
        if 'imply' in expr and 'action' in expr['imply'][0]:
            agent = expr['imply'][0]['action'][0]
            action = expr['imply'][0]['action'][1]
            legal_tree = self._get_pwl_tree(expr['imply'][1], True, False)  # get condition expression as if tree
            weights, threshold, comp = legal_tree['if']
            legal_tree = {'if': KeyedPlane(KeyedVector(weights), threshold, comp),
                          True: legal_tree[True],
                          False: legal_tree[False]}
            legal_tree = makeTree(legal_tree)
            return agent, action, legal_tree.desymbolize(self.world.symbols)

        return None  # could not find action legality constraint in the expression
