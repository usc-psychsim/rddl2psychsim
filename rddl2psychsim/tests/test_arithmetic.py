import unittest
from psychsim.pwl import WORLD
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class TestNumArithmetics(unittest.TestCase):

    def test_sum(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                q : { state-fluent,  int, default = -5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p + q; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(p, 1 - 5)
        self.assertEqual(q, -5)

    def test_sum_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p + -5; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1 - 5)

    def test_sum_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 2 + 5; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 2 + 5)

    def test_sub(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                q : { state-fluent,  int, default = 5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p - q; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(p, 1 - 5)
        self.assertEqual(q, 5)

    def test_sub_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p - 5; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1 - 5)

    def test_sub_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 2 - 5; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 2 - 5)

    def test_mul_invalid(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                q : { state-fluent,  int, default = -5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p * q; }; // cannot multiply variables
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        with self.assertRaises(ValueError):
            conv.convert_str(rddl)

    def test_mul_const(self):
        rddl = '''
           domain my_test {
               pvariables { 
                   p : { state-fluent,  int, default = 1 };
                   a : { action-fluent, bool, default = false }; 
               };
               cpfs { p' = p * 5; };
               reward = 0;
           }
           non-fluents my_test_empty { domain = my_test; }
           instance my_test_inst { domain = my_test; init-state { a; }; }
           '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1 * 5)

    def test_mul_const2(self):
        rddl = '''
           domain my_test {
               pvariables { 
                   p : { state-fluent,  int, default = 1 };
                   a : { action-fluent, bool, default = false }; 
               };
               cpfs { p' = 5 * p; };
               reward = 0;
           }
           non-fluents my_test_empty { domain = my_test; }
           instance my_test_inst { domain = my_test; init-state { a; }; }
           '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 5 * 1)

    def test_mul_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 2 * -5; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 2 * -5)

    def test_div_invalid(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                q : { state-fluent,  int, default = -5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p / q; }; // cannot divide variables
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        with self.assertRaises(ValueError):
            conv.convert_str(rddl)

    def test_div_const(self):
        rddl = '''
           domain my_test {
               pvariables { 
                   p : { state-fluent,  int, default = 1 };
                   a : { action-fluent, bool, default = false }; 
               };
               cpfs { p' = p / 5; };
               reward = 0;
           }
           non-fluents my_test_empty { domain = my_test; }
           instance my_test_inst { domain = my_test; init-state { a; }; }
           '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, int(1 / 5))

    def test_div_invalid_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                q : { state-fluent,  int, default = -5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 5 / p; }; // cannot divide over variables
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        with self.assertRaises(ValueError):
            conv.convert_str(rddl)

    def test_div_consts(self):
        rddl = '''
           domain my_test {
               pvariables { 
                   p : { state-fluent,  real, default = 1 };
                   a : { action-fluent, bool, default = false }; 
               };
               cpfs { p' = p / 5; };
               reward = 0;
           }
           non-fluents my_test_empty { domain = my_test; }
           instance my_test_inst { domain = my_test; init-state { a; }; }
           '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1. / 5.)

    def test_div_consts2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 2 / 5; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, int(2 / 5))

    def test_comp(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  int, default = 1 };
                q : { state-fluent,  int, default = -5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = (p + q) - p * 2 + (q / 4); };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(p, int((1 + -5) - 1 * 2 + (-5 / 4)))
        self.assertEqual(q, -5)

    def test_comp_consts(self):
        rddl = '''
           domain my_test {
               pvariables { 
                   p : { state-fluent,  real, default = 1 };
                   a : { action-fluent, bool, default = false }; 
               };
               cpfs { p' = 2 * (-5.1 + 4.5) / 3  - 1 * 0.9; };
               reward = 0;
           }
           non-fluents my_test_empty { domain = my_test; }
           instance my_test_inst { domain = my_test; init-state { a; }; }
           '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1.)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 2 * (-5.1 + 4.5) / 3 - 1 * 0.9)


class TestBoolArithmetics(unittest.TestCase):

    def test_sum(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p + q; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(p, False)
        self.assertEqual(q, False)

    def test_sum_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p + 5; };
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
        self.assertEqual(p, True)

    def test_sum_const2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p - 5; };
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
        self.assertEqual(p, bool((1 - 5) > 0))  # in PS boolean var is True iff value > 0


if __name__ == '__main__':
    unittest.main()
