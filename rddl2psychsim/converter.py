import logging
from typing import Dict

from pyrddl.rddl import RDDL
from pyrddl.expr import Expression

from psychsim.agent import Agent
from psychsim.world import World
from psychsim.pwl import KeyedVector, CONSTANT, rewardKey, makeTree, makeFuture, KeyedMatrix, KeyedTree, KeyedPlane, \
    setToConstantMatrix, actionKey

from rddl2psychsim.rddl import parse_rddl

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class Converter(object):

    def __init__(self):
        self.model: RDDL = None
        self.world: World = None
        self.fluent_to_feature = {}
        self.constants = {}
        self.actions = {}

    def convert(self, rddl_file: str, agent_name='Agent', verbose=True) -> None:
        # parse RDDL file
        self.model = parse_rddl(rddl_file, verbose)
        domain = self.model.domain

        logging.info('==================================================')
        logging.info(f'Converting RDDL domain "{domain.name}" to PsychSim...')

        # create world and agent #TODO read agent(s) name(s) from RDDL?
        logging.info('__________________________________________________')
        self.world = World()
        agent = self.world.addAgent(agent_name)
        logging.info(f'Created agent {agent.name}')

        # first set value of non-fluents from definition
        logging.info('__________________________________________________')
        self.constants = {}
        if hasattr(self.model.non_fluents, 'init_non_fluent'):
            for nf, val in self.model.non_fluents.init_non_fluent:
                self.constants[nf] = val
                logging.info(f'Initialized constant "{nf}" with value "{val}"')

        # try to initialize non-fluents from definition's default value
        for nf in domain.non_fluents.values():
            if nf.arity != 0:  # can't initialize parameterizable constants
                continue
            nf_name = (nf.name, None)
            if nf_name not in self.constants:  # non-fluent definition on file takes precedence
                self.constants[nf_name] = nf.default
                logging.info(f'Initialized constant "{nf_name}" with default value "{nf.default}"')
        logging.info(f'Total {len(self.constants)} constants initialized: {self.constants}')

        # create variables from state fluents
        logging.info('__________________________________________________')
        self.fluent_to_feature = {}
        for sf in domain.state_fluents.values():
            f_name = f'{sf.name}'
            f = self.world.defineState(agent.name, f_name, type(sf.default))
            self.world.setFeature(f, sf.default)
            self.fluent_to_feature[sf.name] = f
            logging.info(f'Created feature "{f}" from state fluent "{sf.name}"')
        logging.info(f'Total {len(self.fluent_to_feature)} features created')

        # create actions for agent
        logging.info('__________________________________________________')
        self.actions = {agent_name: {}}
        for act in domain.action_fluents.values():
            action = agent.addAction({'verb': act.name})
            self.actions[agent_name][act.name] = action
            logging.info(f'Created action "{action}" from action fluent: {act}')
        logging.info(f'Total {len(self.actions[agent_name])} actions created for agent "{agent.name}"')

        # create reward function
        logging.info('__________________________________________________')
        tree = self._create_dynamics_tree(rewardKey(agent.name), self.model.domain.reward, agent)
        agent.setReward(tree, 1.)
        logging.info(f'Set agent "{agent.name}" reward to:\n{tree}')

        # create dynamics from conditional probability functions (CPFs)
        logging.info('__________________________________________________')
        for cpf in domain.cpfs[1]:
            f_type = cpf.pvar[0]
            if f_type == 'pvar_expr':
                # todo n-arity
                name = cpf.pvar[1][0].replace('\'', '')
                if not self._is_feature(name):
                    raise ValueError(f'Could not find feature for fluent "{name}" in CPF "{cpf}"!')
                f = self._get_feature(name)
                tree = self._create_dynamics_tree(f, cpf.expr, agent)
                self.world.setDynamics(f, True, tree)
                logging.info(f'Set dynamics for feature "{f}" to:\n{tree}')
            else:
                raise NotImplementedError(f'Cannot convert CPF "{cpf}" of type "{f_type}"!')

        self.world.setOrder([{agent.name}])
        logging.info('Done!')

    def _is_feature(self, sf: str) -> bool:
        # todo n-arity
        return sf in self.fluent_to_feature

    def _get_feature(self, sf: str) -> str:
        # todo n-arity
        return self.fluent_to_feature[sf]

    def _is_action(self, af: str, agent: Agent) -> bool:
        # todo n-arity
        return agent.name in self.actions and af in self.actions[agent.name]

    def _get_action(self, af: str, agent: Agent) -> str:
        # todo n-arity
        return self.actions[agent.name][af]

    def _is_constant(self, nf: str) -> bool:
        return any(c[0] == nf for c in self.constants.keys())

    def _get_constant_value(self, nf: str) -> object:
        # todo n-arity
        return next(val for c, val in self.constants.items() if c[0] == nf)

    def _create_dynamics_tree(self, key, expression: Expression, agent: Agent) -> KeyedTree:
        return makeTree(self._get_dynamics_tree(key, self._get_expression_dict(expression, agent)))

    def _get_dynamics_tree(self, key, tree_dict) -> KeyedMatrix or Dict:
        if all(isinstance(k, str) for k in tree_dict.keys()) and all(isinstance(v, float) for v in tree_dict.values()):
            # if all key-value pairs, create a keyed vector
            return KeyedMatrix({makeFuture(key): KeyedVector(tree_dict)})

        if 'if' in tree_dict:
            # build if-then-else tree
            weights, threshold, comp = tree_dict['if']
            return {'if': KeyedPlane(KeyedVector(weights), threshold, comp),
                    True: self._get_dynamics_tree(key, tree_dict[True]),
                    False: self._get_dynamics_tree(key, tree_dict[False])}

        if 'distribution' in tree_dict:
            # create stochastic effect
            return {'distribution': [(setToConstantMatrix(key, v), p) for v, p in tree_dict['distribution']]}

        # KeyedPlane(KeyedVector({key: 1}), value, 0)
        raise NotImplementedError(f'Could not parse RDDL expression, got invalid tree: "{tree_dict}"!')

    def _get_expression_dict(self, expression: Expression, agent: Agent) -> Dict:

        # process leaf node, try to get feature name or constant value
        e_type = expression.etype[0]
        if e_type == 'constant':
            try:
                return {CONSTANT: float(expression.args)}
            except ValueError as e:
                logging.info(f'Could not convert value "{expression.args}" to float in RDDL expression "{expression}"!')
                raise e

        if e_type == 'pvar':
            name = expression.args[0]
            if self._is_feature(name):
                return {self._get_feature(name): 1.}

            if self._is_action(name, agent):
                return {'action': self._get_action(name, agent)}

            if self._is_constant(name):
                try:
                    value = self._get_constant_value(name)
                    return {CONSTANT: float(value)}
                except ValueError as e:
                    logging.info(f'Could not convert value "{value}" to float in RDDL expression "{expression}"!')
                    raise e

            raise ValueError(f'Could not find feature, action or constant from RDDL expression "{expression}"!')

        # process arithmetic node
        if e_type == 'arithmetic':
            def _get_const_val(s):
                return s if isinstance(s, float) else s[CONSTANT] if len(s) == 1 and CONSTANT in s else None

            def _update_weights(s):
                for k, v in s.items():
                    assert isinstance(k, str) and isinstance(v, float), \
                        f'Could not parse RDDL expression "{expression}", invalid nested PWL arithmetic in "{s}"!'
                    weights[k] = weights[k] + v if k in weights else v  # add weight if key already in dict

            lhs = self._get_expression_dict(expression.args[0], agent)
            rhs = self._get_expression_dict(expression.args[1], agent)
            lhs_const = _get_const_val(lhs)
            rhs_const = _get_const_val(rhs)
            all_consts = isinstance(lhs_const, float) and isinstance(rhs_const, float)

            weights = {}
            a_type = expression.etype[1]
            if a_type == '+':
                if all_consts:
                    weights.update({CONSTANT: lhs_const + rhs_const})  # reduce
                else:
                    _update_weights(lhs)  # if addition, just add everything from both sides
                    _update_weights(rhs)
            elif a_type == '-':
                if all_consts:
                    weights.update({CONSTANT: lhs_const - rhs_const})  # reduce
                else:
                    _update_weights(lhs)  # if subtraction, first get left-hand side
                    _update_weights({k: -v for k, v in rhs.items()})  # then multiply right-hand side by -1
            elif a_type == '*':
                # if multiplication, only works if one or both sides are constants
                if all_consts:
                    weights.update({CONSTANT: lhs_const * rhs_const})  # reduce
                elif isinstance(lhs_const, float):
                    _update_weights({k: lhs_const * v for k, v in rhs.items()})  # multiply right-hand side by const
                elif isinstance(rhs_const, float):
                    _update_weights({k: rhs_const * v for k, v in lhs.items()})  # multiply left-hand side by const
                else:
                    raise ValueError(f'Non-PWL operation not supported: "{expression}"!')
            elif a_type == '/':
                # if division, only works if right or both sides are constants
                if all_consts:
                    weights.update({CONSTANT: lhs_const / rhs_const})  # reduce
                elif isinstance(rhs_const, float):
                    _update_weights({k: v / rhs_const for k, v in lhs.items()})  # divide left-hand side by const
                else:
                    raise ValueError(f'Non-PWL operation not supported: "{expression}"!')
            else:
                NotImplementedError(f'Cannot parse arithmetic expression: "{expression}" of type "{a_type}"!')
            return weights

        if e_type == 'boolean':
            def _get_const_val(s):
                return s if isinstance(s, float) else s[CONSTANT] if len(s) == 1 and CONSTANT in s else None

            def _update_weights(s):
                for k, v in s.items():
                    assert isinstance(k, str) and isinstance(v, float), \
                        f'Could not parse RDDL expression "{expression}", invalid nested PWL AND boolean in "{s}"!'
                    if v == 0:
                        weights.clear()
                        weights[CONSTANT] = 0.  # if a value of 0 is associated with feature, then False (0)
                        return

                    weights[k] = 1. if v > 0 else -1.  # truncate weight to 1 / -1 to allow later comparisons / planes

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
                    return lhs if bool(rhs_const) else {CONSTANT: 0.}
                if isinstance(lhs_const, float):
                    return rhs if bool(lhs_const) else {CONSTANT: 0.}
                if all(isinstance(v, float) for v in lhs.values()) and all(isinstance(v, float) for v in rhs.values()):
                    _update_weights(lhs)  # if both vectors, just add everything from both sides
                    _update_weights(rhs)
                    return weights
                return {'^': (lhs, rhs)}

            if b_type == '|':
                # if OR, one side has to be True
                if all_consts:
                    return {CONSTANT: float(bool(rhs_const) or bool(lhs_const))}
                if isinstance(rhs_const, float):
                    return {CONSTANT: 1.} if bool(rhs_const) else lhs
                if isinstance(lhs_const, float):
                    return {CONSTANT: 1.} if bool(lhs_const) else rhs
                return {'|': (lhs, rhs)}

            if b_type == '~':
                # if NOT, multiply weights by -1
                if isinstance(lhs_const, float):
                    return {CONSTANT: 0. if bool(lhs_const) else 1.}
                if all(isinstance(v, float) for v in lhs.values()):
                    return {k: -v for k, v in lhs.items()}
                return {'~': lhs}

            if b_type == '<=>':
                # if EQUIV, sides have to be of equal value
                if all_consts:
                    return {CONSTANT: float(rhs_const == lhs_const)}
                if rhs == lhs:
                    return {CONSTANT: 1.}

                lhs = list(lhs.items())
                rhs = list(rhs.items())
                assert len(lhs) == 1 and isinstance(lhs[0][1], float) and \
                       len(rhs) == 1 and isinstance(rhs[0][1], float), \
                    f'Could not parse boolean expression "{expression}", invalid PWL equivalence composition!'
                return {'<=>': {lhs[0][0]: lhs[0][1], rhs[0][0]: -lhs[0][1]}}

            if b_type == '=>':
                # if IMPLICATION, false only if left is true and right is false
                if all_consts:
                    return {CONSTANT: float(bool(rhs_const) or not bool(lhs_const))}
                if isinstance(lhs_const, float):
                    if not bool(lhs_const):
                        return {CONSTANT: 1.}  # left is false, so implication is true
                    return rhs  # left is true, so right has to be true
                if isinstance(rhs_const, float):
                    if bool(rhs_const):
                        return {CONSTANT: 1.}  # right is true, so implication is true
                    if all(isinstance(v, float) for v in lhs.values()):
                        return {k: -v for k, v in lhs.items()}  # right is false, negate left
                    return {'~': lhs}

                lhs = list(lhs.items())
                rhs = list(rhs.items())
                assert len(lhs) == 1 and isinstance(lhs[0][1], float) and \
                       len(rhs) == 1 and isinstance(rhs[0][1], float), \
                    f'Could not parse boolean expression "{expression}", invalid PWL equivalence composition!'
                return {'<=>': {lhs[0][0]: lhs[0][1], rhs[0][0]: -lhs[0][1]}}

            raise NotImplementedError(f'Cannot parse boolean expression: "{expression}" of type "{b_type}"!')

        def _nested_if(c, tb, fb):
            if all(isinstance(v, float) for v in c.values()):
                # takes AND of features in vector
                return {'if': (c, len([v for v in c.values() if v > 0]) - 0.5, 1),
                        True: tb,
                        False: fb}
            if '<=>' in c:
                # takes equality of features in vector
                return {'if': (c['<=>'], 0, 0),
                        True: tb,
                        False: fb}
            if 'action' in c and len(c) == 1:
                # special conditional on agent's action
                return {'if': ({actionKey(agent.name): 1.}, c['action'], 0),
                        True: tb,
                        False: fb}
            if '^' in c and len(c) == 1:
                lhs, rhs = c['^']  # composes nested AND tree
                return {'if': lhs,
                        True: _nested_if(rhs, tb, fb),
                        False: fb}
            if '|' in c and len(c) == 1:
                lhs, rhs = c['|']  # composes nested OR tree
                return {'if': lhs,
                        True: tb,
                        False: _nested_if(rhs, tb, fb)}
            if '~' in c and len(c) == 1:
                return _nested_if(c['~'], fb, tb)  # if NOT, just flip branches

            raise ValueError(f'Could not parse RDDL expression "{expression}", '
                             f'invalid nested PWL control in "{c}"!')

        if e_type == 'control':

            c_type = expression.etype[1]
            if c_type == 'if':
                # get condition and branches
                cond = self._get_expression_dict(expression.args[0], agent)
                true_branch = self._get_expression_dict(expression.args[1], agent)
                false_branch = self._get_expression_dict(expression.args[2], agent)
                return _nested_if(cond, true_branch, false_branch)
            else:
                raise NotImplementedError(f'Cannot parse control expression: "{expression}" of type "{c_type}"!')

        if e_type == 'randomvar':

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
                return _nested_if(arg, {CONSTANT: 1.}, {CONSTANT: 0.})

            else:
                raise NotImplementedError(f'Cannot parse stochastic expression: "{expression}" of type "{d_type}"!')

        # not yet implemented
        raise NotImplementedError(f'Cannot parse expression: "{expression}" of type "{e_type}"!')
