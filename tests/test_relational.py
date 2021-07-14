import unittest
from psychsim.pwl import WORLD
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class TestRelational(unittest.TestCase):

    def test_eq_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  int, default = 4 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q == r; };
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

    def test_eq_false_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = -1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q == 3; };
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

    def test_eq_false_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 1.0 == 3; };
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

    def test_eq_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q == r; };
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

    def test_eq_true_num(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  real, default = 1.0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q == r; };
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

    def test_eq_true_self(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q == q; };
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

    def test_eq_true_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = -1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q == -1; };
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

    def test_eq_true_vars_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = -1 };
                r : { state-fluent,  int, default = 5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q + r == -1 + r; };
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

    def test_eq_true_num_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 1.0 == 1; };
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

    def test_eq_true_bool_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = false == false; };
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

    def test_eq_true_linear(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  int, default = 4 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q * 3 + r - 1 == - 1 + q + 2 * r + q - r + q; };
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

    def test_neq_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  real, default = 1.0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q ~= r; };
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

    def test_neq_false_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = -1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q ~= -1; };
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

    def test_neq_false_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 1.0 ~= 1; };
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

    def test_neq_false_self(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q ~= q; };
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

    def test_neq_false_vars_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = -1 };
                r : { state-fluent,  int, default = 5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q + r ~= -1 + r; };
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

    def test_neq_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  int, default = -1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q ~= r; };
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

    def test_neq_true_num(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  real, default = 1.0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q ~= r; };
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

    def test_neq_true_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = -1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q ~= 1; };
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

    def test_neq_true_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = -1.0 ~= 1; };
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

    def test_geq_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  int, default = 4 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q >= r; };
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

    def test_geq_false_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = -1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q >= 3; };
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

    def test_geq_false_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 1.0 >= 3; };
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

    def test_geq_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  int, default = -1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q >= r; };
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

    def test_geq_true_num(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  real, default = 1.0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q >= r; };
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

    def test_geq_true_self(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q >= q; };
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

    def test_geq_true_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  real, default = 1.0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q >= -1; };
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

    def test_geq_true_vars_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = -1 };
                r : { state-fluent,  int, default = 5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q + r >= -1 + r; };
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

    def test_geq_true_num_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 2.0 >= 1; };
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

    def test_geq_true_bool_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = false >= false; };
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

    def test_geq_true_linear(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  int, default = 4 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q * 3 + r - 1 >= - 1 + q + 2 * r + q - r + q; };
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

    def test_leq_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 4 };
                r : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q <= r; };
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

    def test_leq_false_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = -1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q <= -3; };
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

    def test_leq_false_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 1.0 <= -3; };
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

    def test_leq_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = -1 };
                r : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q <= r; };
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

    def test_leq_true_num(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  real, default = 1.0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q <= r; };
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

    def test_leq_true_self(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q <= q; };
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

    def test_leq_true_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  real, default = -1.0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q <= 1; };
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

    def test_leq_true_vars_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = -1 };
                r : { state-fluent,  int, default = 5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q + r <= -1 + r; };
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

    def test_leq_true_num_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = -2.0 <= 1; };
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

    def test_leq_true_bool_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = false <= false; };
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

    def test_leq_true_linear(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  int, default = 4 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q * 3 + r - 1 <= - 1 + q + 2 * r + q - r + q; };
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

    def test_gt_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  real, default = 1.0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q > r; };
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

    def test_gt_false_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = -2 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q > -1; };
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

    def test_gt_false_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 1.0 > 1; };
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

    def test_gt_false_self(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q > q; };
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

    def test_gt_false_vars_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = -1 };
                r : { state-fluent,  int, default = 5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q + r > -1 + r; };
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

    def test_gt_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  int, default = -1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q > r; };
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

    def test_gt_true_num(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 2 };
                r : { state-fluent,  real, default = 1.0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q > r; };
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

    def test_gt_true_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q > -1; };
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

    def test_gt_true_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 1.0 > -1; };
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

    def test_gt_true_vars_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = -1 };
                r : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q + r > -1; };
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

    def test_lt_false(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 1 };
                r : { state-fluent,  real, default = 1.0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q < r; };
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

    def test_lt_false_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 2 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q < -1; };
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

    def test_lt_false_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = 1.0 < 1; };
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

    def test_lt_false_self(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q < q; };
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

    def test_lt_false_vars_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = true };
                q : { state-fluent,  int, default = -1 };
                r : { state-fluent,  int, default = 5 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q + r < -1 + r; };
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

    def test_lt_true(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = -1 };
                r : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q < r; };
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

    def test_lt_true_num(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = -2 };
                r : { state-fluent,  real, default = 1.0 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q < r; };
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

    def test_lt_true_const(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q < 3.0; };
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

    def test_lt_true_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = -3.0 < -2; };
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

    def test_lt_true_vars_consts(self):
        rddl = '''
        domain my_test {
            pvariables { 
                p : { state-fluent,  bool, default = false };
                q : { state-fluent,  int, default = -1 };
                r : { state-fluent,  int, default = 1 };
                a : { action-fluent, bool, default = false }; 
            };
            cpfs { p' = q + r - 1 < -1 + r; };
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


if __name__ == '__main__':
    unittest.main()
