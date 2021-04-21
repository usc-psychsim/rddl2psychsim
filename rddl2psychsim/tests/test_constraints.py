import unittest
from psychsim.pwl import WORLD, actionKey
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class TestConstraints(unittest.TestCase):

    def test_state(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = p + 1; };
                    reward = 0;
                    state-action-constraints { p >= 0; };
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.verify_constraints()
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.verify_constraints()

    def test_state_false(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = p + 1; };
                    reward = 0;
                    state-action-constraints { p < 0; };
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        with self.assertRaises(AssertionError):
            conv.verify_constraints()

    def test_const(self):
        rddl = '''
                domain my_test {
                    pvariables {
                        C : { non-fluent, int, default = 1}; 
                        p : { state-fluent,  int, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = p + C; };
                    reward = 0;
                    state-action-constraints { C >= 0; };
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.verify_constraints()
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.verify_constraints()

    def test_const_false(self):
        rddl = '''
                domain my_test {
                    pvariables {
                        C : { non-fluent, int, default = 1}; 
                        p : { state-fluent,  int, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = p + C; };
                    reward = 0;
                    state-action-constraints { C < 1; }; // will be asserted at conversion time
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter(const_as_assert=True)
        with self.assertRaises(AssertionError):
            conv.convert_str(rddl)

    def test_action(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = p + 1; };
                    reward = 0;
                    state-action-constraints { a; };
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.verify_constraints()

    def test_action_false(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = p + 1; };
                    reward = 0;
                    state-action-constraints { ~a; };   // action is always true 
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        with self.assertRaises(AssertionError):
            conv.verify_constraints()

    def test_actions_legal(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        q : { state-fluent,  int, default = 1 };
                        a1 : { action-fluent, bool, default = false }; 
                        a2 : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (a1') then
                                    p + 2
                                else if (a2') then
                                    p - 2
                                else
                                    100;
                    };
                    reward = p;
                    state-action-constraints { a1 => q > 1; a2 => q <= 1; }; 
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a1; }; horizon  = 3; }
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
        self.assertEqual(p, -2)


if __name__ == '__main__':
    unittest.main()
