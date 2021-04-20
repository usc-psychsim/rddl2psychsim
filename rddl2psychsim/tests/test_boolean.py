import unittest
from psychsim.pwl import WORLD
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class TestBoolean(unittest.TestCase):

    def test_and_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p ^ q; };
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
        self.assertEqual(q, True)

    def test_and_false2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p ^ p ^ q; };
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

    def test_and_false_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p ^ 0; };
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
        self.assertEqual(p, False)

    def test_and_false_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 1 ^ 0; };
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
        self.assertEqual(p, False)

    def test_and_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p ^ q; };
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
        self.assertEqual(p, True)
        self.assertEqual(q, True)

    def test_and_true_not(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p ^ ~q; };
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
        self.assertEqual(p, True)
        self.assertEqual(q, False)

    def test_and_true2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  bool, default = true };
                r : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p ^ q ^ r; };
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
        r = conv.world.getState(WORLD, 'r', unique=True)
        self.assertEqual(p, True)
        self.assertEqual(q, True)
        self.assertEqual(r, True)

    def test_and_false_num(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent, bool, default = false };
                q : { state-fluent, real, default = 0.5 };
                r : { state-fluent, int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q ^ r; };   // sum needed to be > 1.5
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
        r = conv.world.getState(WORLD, 'r', unique=True)
        self.assertEqual(p, False)
        self.assertEqual(q, 0.5)
        self.assertEqual(r, 1)

    def test_and_false_num2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent, bool, default = false };
                q : { state-fluent, real, default = 0.5 };
                r : { state-fluent, int, default = 0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 4 * q ^ r; };   // sum needed to be > (4 * 1 + 1) - 0.5
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
        r = conv.world.getState(WORLD, 'r', unique=True)
        self.assertEqual(p, False)
        self.assertEqual(q, 0.5)
        self.assertEqual(r, 0)

    def test_and_true_self(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p ^ p; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_and_true_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p ^ 1; };
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

    def test_and_true_const2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p ^ 4.9; };
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

    def test_and_true_const3(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p ^ -1; };
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

    def test_and_true_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 3 ^ 1; };
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

    def test_or_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p | q; };
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

    def test_or_false2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p | p | q; };
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

    def test_or_false_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p | 0; };
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

    def test_or_false_const2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 0 | 0; };
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
        self.assertEqual(p, False)

    def test_or_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p | q; };
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
        self.assertEqual(p, True)
        self.assertEqual(q, False)

    def test_or_true2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = false };
                r : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p | q | r; };
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
        r = conv.world.getState(WORLD, 'r', unique=True)
        self.assertEqual(p, True)
        self.assertEqual(q, False)
        self.assertEqual(r, True)

    def test_or_true_not(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p | ~q; };
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
        self.assertEqual(p, True)
        self.assertEqual(q, False)

    def test_or_true_self(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p | p; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_or_true_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p | 1; };
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

    def test_or_true_const2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p | 4.9; };
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

    def test_or_true_const3(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p | -1; };
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

    def test_or_true_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 6 | 0; };
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

    def test_not_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = ~p; };
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
        self.assertEqual(p, False)

    def test_not_false_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = ~4.2; };
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
        self.assertEqual(p, False)

    def test_not_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = ~p; };
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

    def test_not_true_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = ~0.0; };
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

    def test_not_and_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = ~(p ^ q); };
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
        self.assertEqual(p, True)
        self.assertEqual(q, True)

    def test_not_and_true2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = ~(p ^ q); };
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
        self.assertEqual(p, True)
        self.assertEqual(q, False)

    def test_not_or_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = ~(p | q); };
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
        self.assertEqual(p, True)
        self.assertEqual(q, False)

    def test_not_or_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = ~(p | q); };
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

    def test_not_not_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = ~ ~p; };
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

    def test_comb_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = (~(p ^ q) | false) ^ ~false; };
            reward = 0;
        }
        non-fluents my_test_empty { domain = my_test; }
        instance my_test_inst { domain = my_test; init-state { a; }; }
        '''
        conv = Converter()
        conv.convert_str(rddl)
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(p_, False)
        self.assertEqual(q, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, (not (p and q) or False) and not False)

    def test_equiv_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p <=> q; };
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
        self.assertEqual(q, True)

    def test_equiv_false2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p <=> q; };
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

    def test_equiv_false_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p <=> 0.0; };
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
        self.assertEqual(p, False)

    def test_equiv_false_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 0 <=> 1.0; };
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
        self.assertEqual(p, False)

    def test_equiv_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p <=> q; };
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
        self.assertEqual(p, True)
        self.assertEqual(q, True)

    def test_equiv_true2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p <=> q; };
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
        self.assertEqual(p, True)
        self.assertEqual(q, False)

    def test_equiv_true_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p <=> 0.0; };
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

    def test_equiv_true_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = -1 <=> 1.0; };  // both have true boolean value
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

    def test_equiv_self(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p <=> p; };
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

    def test_equiv_triple(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = false };
                r : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = (p <=> q) <=> r; };
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
        self.assertEqual(p, False)
        self.assertEqual(q, False)
        self.assertEqual(r, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_equiv_rel(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = (q >= 2) <=> (q < 0); };
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
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(p, True)
        self.assertEqual(q, 1)

    def test_imply_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p => q; };
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

    def test_imply_false_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p => 0.0; };
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
        self.assertEqual(p, False)

    def test_imply_false_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = -1 => 0.0; };
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
        self.assertEqual(p, False)

    def test_imply_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p => q; };
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
        self.assertEqual(p, True)
        self.assertEqual(q, True)

    def test_imply_true2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p => q; };
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
        self.assertEqual(p, True)
        self.assertEqual(q, False)

    def test_imply_true3(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p => q; };
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
        self.assertEqual(p, True)
        self.assertEqual(q, True)

    def test_imply_true_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p => 1.0; };
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

    def test_imply_true_const2(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = p => 0; };
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

    def test_imply_true_const3(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 0 => p; };
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

    def test_imply_true_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = -1 => 1.0; };
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

    def test_imply_triple(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  bool, default = true };
                r : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = (p => q) => r; }; 
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
        self.assertEqual(p, False)
        self.assertEqual(q, True)
        self.assertEqual(r, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)

    def test_imply_rel(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 2 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = (q >= 2) => (q < 0); };
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
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(p, False)
        self.assertEqual(q, 2)


if __name__ == '__main__':
    unittest.main()
