import unittest
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

AG_NAME = 'Agent'


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
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, True)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
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
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
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
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        q = conv.world.getState(AG_NAME, 'q', unique=True)
        r = conv.world.getState(AG_NAME, 'r', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, q if (q > r) else r)

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
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 6 if (2 != 4) else 3)

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
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        q = conv.world.getState(AG_NAME, 'q', unique=True)
        r = conv.world.getState(AG_NAME, 'r', unique=True)
        s = conv.world.getState(AG_NAME, 's', unique=True)
        self.assertEqual(p, _python_switch(p + 2, {q: 1,
                                                   'low': 2,
                                                   3: s,
                                                   3 + r: 3,
                                                   'default': 4 + 4}))


if __name__ == '__main__':
    unittest.main()
