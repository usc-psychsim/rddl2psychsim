import unittest
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

AG_NAME = 'Agent'


class TestTypes(unittest.TestCase):

    def test_kron_delta(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = KronDelta(2); }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 2)

    def test_dirac_delta(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  real, default = 1.4 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = DiracDelta(2.5); }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 1.4)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 2.5)

    def test_bernoulli(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  real, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = Bernoulli(.3); }; 
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
        p = conv.world.getState(AG_NAME, 'p')
        self.assertEqual(p[0], 0.7)
        self.assertEqual(p[1], 0.3)

    def test_discrete_non_const(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high}; 
                    };
                    pvariables { 
                        p : { state-fluent,  enum_level, default = @low };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { 
                        p' = Discrete(enum_level,
                                @low : if (p >= 2) then 0.5 else 0.2,
                                @medium : if (p >= 2) then 0.2 else 0.5,
                                @high : 0.3
                            );
                    }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        with self.assertRaises(AssertionError):
            conv.convert_str(rddl, AG_NAME)

    def test_discrete_num(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high}; 
                    };
                    pvariables { 
                        p : { state-fluent,  enum_level, default = @low };
                        q : { state-fluent,  int, default = 3 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { 
                        p' = if (q >= 2) then
                                Discrete(enum_level, @low : 0.5, @medium : 0.2, @high : 0.3)
                            else
                                Discrete(enum_level, @low : 0.2, @medium : 0.5, @high : 0.3);
                    }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 'low')
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p')
        self.assertEqual(p['low'], 0.5)
        self.assertEqual(p['medium'], 0.2)
        self.assertEqual(p['high'], 0.3)
