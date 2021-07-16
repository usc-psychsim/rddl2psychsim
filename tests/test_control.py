import unittest
from psychsim.pwl import WORLD, actionKey, stateKey
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def _python_switch(cond, switch):
    for val, ret in switch.items():
        if cond == val:
            return ret
    return switch['default']


class TestRelational(unittest.TestCase):

    def test_if_true(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  bool, default = true };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (p) then true else false; };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_if_false(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  bool, default = false };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (p) then true else false; };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)

    def test_if_vars(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        q : { state-fluent,  int, default = 1 };
                        r : { state-fluent,  int, default = 4 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (q > r) then q else r; };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        r = conv.world.getState(WORLD, 'r', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, q if (q > r) else r)

    def test_if_action(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        q : { state-fluent,  int, default = -3 };
                        r : { state-fluent,  int, default = 4 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (a) then q else r; };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        r = conv.world.getState(WORLD, 'r', unique=True)
        self.assertEqual(p, 0)
        self.assertEqual(q, -3)
        self.assertEqual(r, 4)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, q)

    def test_if_action_choice(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        q : { state-fluent,  int, default = 1 };
                        a1 : { action-fluent, bool, default = false }; 
                        a2 : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (a1) then
                                    p + 1
                                else if (a2) then
                                    p - 1
                                else
                                    9;
                    };
                    reward = -p;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a1; }; horizon  = 2; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        ag_name = next(iter(conv.world.agents.keys()))
        a = conv.world.getFeature(actionKey(ag_name), unique=True)
        self.assertEqual(a, conv.actions[ag_name]['a2'])
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, -1)

    def test_if_action_dynamics(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        q : { state-fluent,  int, default = 1 };
                        a1 : { action-fluent, bool, default = false }; 
                        a2 : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (a1) then
                                    p + 1
                                else if (a2) then
                                    p - 1
                                else
                                    0;
                    };
                    reward = -p;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a1; }; horizon  = 2; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        conv.world.step()
        ag_name = next(iter(conv.world.agents.keys()))
        a1 = conv.actions[ag_name]['a1']
        a2 = conv.actions[ag_name]['a2']
        p = stateKey(WORLD, 'p')
        self.assertIn(a1, conv.world.dynamics)
        self.assertIn(a2, conv.world.dynamics)
        self.assertIn(True, conv.world.dynamics)
        self.assertIn(p, conv.world.dynamics[a1])
        self.assertIn(p, conv.world.dynamics[a2])
        self.assertIn(p, conv.world.dynamics[True])
        self.assertIn(a1, conv.world.dynamics[p])
        self.assertIn(a2, conv.world.dynamics[p])
        self.assertIn(True, conv.world.dynamics[p])

    def test_if_enum(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high};
                    };
                    pvariables { 
                        p : { state-fluent,  enum_level, default = @low };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (p == @medium) then @high else @medium; };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        _p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(_p, 'low')
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 'high' if (_p == 'medium') else 'medium')

    def test_if_not_enum(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high};
                    };
                    pvariables { 
                        p : { state-fluent,  enum_level, default = @low };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (p ~= @medium) then @high else @medium; };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        _p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(_p, 'low')
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 'high' if (_p != 'medium') else 'medium')

    def test_if_nested_rel(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = -1 };
                        q : { state-fluent,  int, default = 5 };
                        r : { state-fluent,  int, default = 4 };
                        s : { state-fluent,  int, default = 6 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (q > r) then if (q > s) then q else 0 else -1; };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        r = conv.world.getState(WORLD, 'r', unique=True)
        s = conv.world.getState(WORLD, 's', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        if q > r:
            if q > s:
                p_ = q
            else:
                p_ = 0
        else:
            p_ = -1
        self.assertEqual(p, p_)

    def test_if_nested_bool(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = -1 };
                        q : { state-fluent,  bool, default = false };
                        r : { state-fluent,  bool, default = true };
                        s : { state-fluent,  bool, default = false };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (~q ^ r) then if (s | (p < 0)) then 2 * p else 0 else -1; };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        _p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        r = conv.world.getState(WORLD, 'r', unique=True)
        s = conv.world.getState(WORLD, 's', unique=True)
        self.assertEqual(_p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        if not q and r:
            if s or (p < 0):
                p_ = 2 * _p
            else:
                p_ = 0
        else:
            p_ = -1
        self.assertEqual(p, p_)

    def test_if_const(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (2 ~= 4) then 6 else 3; };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 6 if (2 != 4) else 3)

    def test_switch_vars(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        q : { state-fluent,  int, default = 1 };
                        r : { state-fluent,  int, default = 2 };
                        s : { state-fluent,  int, default = 3 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = switch (p) {
                                    case q : q,
                                    case r : r,
                                    case s : s,
                                    default: 4
                                }; 
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        _p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(_p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        r = conv.world.getState(WORLD, 'r', unique=True)
        s = conv.world.getState(WORLD, 's', unique=True)
        self.assertEqual(p, _python_switch(_p, {q: q,
                                                r: r,
                                                s: s,
                                                'default': 4}))

    def test_switch_nested_if(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        q : { state-fluent,  int, default = 1 };
                        r : { state-fluent,  int, default = 4 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = switch (p) {
                                    case q : if (q > r) then q else r,
                                    case r: r,
                                    default: 4
                                }; 
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        _p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(_p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        r = conv.world.getState(WORLD, 'r', unique=True)
        self.assertEqual(p, _python_switch(_p, {q: q if (q > r) else r,
                                                r: r,
                                                'default': 4}))

    def test_switch_consts(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = switch (p) {
                                    case 1 : 1,
                                    case 2 : 2,
                                    case 3 : 3,
                                    default: 4
                                }; 
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        _p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(_p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, _python_switch(_p, {1: 1,
                                                2: 2,
                                                3: 3,
                                                'default': 4}))

    def test_switch_no_case(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = switch (p) {
                                    default : 1
                                }; 
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)

    def test_switch_cond_op(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        q : { state-fluent,  int, default = 1 };
                        r : { state-fluent,  int, default = 2 };
                        s : { state-fluent,  int, default = 3 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = switch (q + 2 * r) {
                                    case q : q,
                                    case r : r,
                                    case s : s,
                                    default: 4
                                }; 
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        r = conv.world.getState(WORLD, 'r', unique=True)
        s = conv.world.getState(WORLD, 's', unique=True)
        self.assertEqual(p, _python_switch(q + 2 * r, {q: q,
                                                       r: r,
                                                       s: s,
                                                       'default': 4}))

    def test_switch_case_op(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        q : { state-fluent,  int, default = 1 };
                        r : { state-fluent,  int, default = 2 };
                        s : { state-fluent,  int, default = 3 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = switch (p) {
                                    case 2 * q - r : q,
                                    case r : r,
                                    case s : s,
                                    default: 4
                                }; 
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        _p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(_p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        r = conv.world.getState(WORLD, 'r', unique=True)
        s = conv.world.getState(WORLD, 's', unique=True)
        self.assertEqual(p, _python_switch(_p, {2 * q - r: q,
                                                r: r,
                                                s: s,
                                                'default': 4}))

    def test_switch_case_if(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high};
                    };
                    pvariables { 
                        p : { state-fluent,  enum_level, default = @low };
                        q : { state-fluent,  enum_level, default = @low };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = switch (q) {
                                    case @low    : if(p==@medium) then @high else @low,
                                    case @medium : if(p==@low) then @medium else @high,
                                    default      : if(p==@high) then @low else @medium
                                }; 
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        _p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(_p, 'low')
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(p, _python_switch(q, {'low': 'high' if p == 'medium' else 'low',
                                               'medium': 'medium' if p == 'low' else 'high',
                                               'default': 'low' if p == 'high' else 'medium'}))

    def test_switch_enum(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high};
                    };
                    pvariables { 
                        p : { state-fluent,  enum_level, default = @low };
                        q : { state-fluent,  enum_level, default = @medium };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = switch (q) {
                                    case @low : @low,
                                    case @medium : @medium,
                                    default : @high
                                }; 
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        _p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(_p, 'low')
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(p, _python_switch(q, {'low': 'low',
                                               'medium': 'medium',
                                               'default': 'high'}))

    def test_switch_no_default(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = switch (p) {
                                    case 2 : 1
                                }; 
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        with self.assertRaises(AssertionError):
            conv.convert_str(rddl)

    def test_switch_multiple_default(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = switch (p) {
                                    default: 3,
                                    case 2 : 1,
                                    default: 2
                                }; 
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        with self.assertRaises(AssertionError):
            conv.convert_str(rddl)

    def test_switch_vars_consts_enum(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high};
                    };
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        q : { state-fluent,  enum_level, default = @high };
                        r : { state-fluent,  int, default = 4 };
                        s : { state-fluent,  int, default = 5 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = switch (p+2) {
                                    case q : 1,
                                    case @low : 2,
                                    case 3 : s,
                                    case 3 + r : 3,
                                    default: 4 + 4
                                }; 
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        r = conv.world.getState(WORLD, 'r', unique=True)
        s = conv.world.getState(WORLD, 's', unique=True)
        self.assertEqual(p, _python_switch(p + 2, {q: 1,
                                                   'low': 2,
                                                   3: s,
                                                   3 + r: 3,
                                                   'default': 4 + 4}))

    def test_if_switch_enum(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high};
                    };
                    pvariables {
                        p : { state-fluent,  enum_level, default = @low };
                        q : { state-fluent,  enum_level, default = @low };
                        a : { action-fluent, bool, default = false };
                    };
                    cpfs { p' = if (switch (q) {
                                        case @low : true,
                                        case @medium : true,
                                        default : false
                                    })
                                then @high
                                else @low;
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        _p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(_p, 'low')
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(p, 'high' if _python_switch(q, {'low': True,
                                                         'medium': True,
                                                         'default': False}) else 'low')

    def test_if_switch_enum_var(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high};
                    };
                    pvariables {
                        p : { state-fluent,  enum_level, default = @low };
                        q : { state-fluent,  enum_level, default = @low };
                        r : { state-fluent,  bool, default = true };
                        a : { action-fluent, bool, default = false };
                    };
                    cpfs { p' = if (switch (q) {
                                        case @low : r,
                                        case @medium : r,
                                        default : ~r
                                    })
                                then @high
                                else @low;
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        _p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(_p, 'low')
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        r = conv.world.getState(WORLD, 'r', unique=True)
        self.assertEqual(p, 'high' if _python_switch(q, {'low': r,
                                                         'medium': r,
                                                         'default': not r}) else 'low')


if __name__ == '__main__':
    unittest.main()
