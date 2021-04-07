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


class TestTypes(unittest.TestCase):

    def test_enum_def(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high};
                    };
                    pvariables { 
                        p : { state-fluent,  enum_level, default = @high };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = @medium; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 'high')
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 'medium')

    def test_enum_init(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high};
                    };
                    pvariables { 
                        p : { state-fluent,  enum_level, default = @high };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = @medium; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { p = @low; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 'low')
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 'medium')
