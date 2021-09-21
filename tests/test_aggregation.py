import unittest
import numpy as np
from psychsim.pwl import WORLD, stateKey
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class TestAggregation(unittest.TestCase):

    def test_fluent_sum(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, int, default = -1 }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = sum_{{?x : obj}}[q(?x)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={v}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, sum(objs.values()))

    def test_fluent_multi_sum(self):
        objs = {('x1', 'x1'): 1,
                ('x1', 'x2'): 2,
                ('x2', 'x1'): 3,
                ('x2', 'x2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        obj : object;
                    }};
                    pvariables {{ 
                        p : {{ state-fluent, int, default = -1 }};
                        q(obj, obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = sum_{{?x : obj, ?y : obj}}[q(?x, ?y)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({",".join(o)}) = {v}' for o, v in objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, sum(objs.values()))

    def test_fluent_multi_sum2(self):
        objs = {('x1', 'y1'): 1,
                ('x1', 'y2'): 2,
                ('x2', 'y1'): 3,
                ('x2', 'y2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_obj : object;
                        y_obj : object; 
                    }};
                    pvariables {{ 
                        p : {{ state-fluent, int, default = -1 }};
                        q(x_obj, y_obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = 2 * sum_{{?x : x_obj, ?y : y_obj}}[q(?x, ?y)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        x_obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                        y_obj : {{{', '.join({y for _, y in objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({",".join(o)}) = {v}' for o, v in objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 2 * sum(objs.values()))

    def test_fluent_multi_sum3(self):
        objs1 = {('x1', 'y1'): 1,
                 ('x1', 'y2'): 2,
                 ('x2', 'y1'): 3,
                 ('x2', 'y2'): 4}
        objs2 = {'x1': 3,
                 'x2': -1}

        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_obj : object;
                        y_obj : object; 
                    }};
                    pvariables {{ 
                        p : {{ state-fluent, int, default = -1 }};
                        q(x_obj, y_obj) : {{ state-fluent, int, default = -1 }};
                        r(x_obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = 2 * sum_{{?x : x_obj, ?y : y_obj}}[ q(?x, ?y) + 2 * r(?x) ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        x_obj : {{{', '.join({x for x, _ in objs1.keys()})}}};
                        y_obj : {{{', '.join({y for _, y in objs1.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({",".join(o)}) = {v}' for o, v in objs1.items())};
                        {'; '.join(f'r({o}) = {v}' for o, v in objs2.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 2 * sum(objs1[(x, y)] + 2 * objs2[x] for x, y in objs1.keys()))

    def test_non_fluent_sum(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        C(obj) : {{ non-fluent, int, default = -1}};
                        p : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = sum_{{?x : obj}}[C(?x)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    non-fluents {{
                        {'; '.join(f'C({o})={v}' for o, v in objs.items())};
                    }};
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, sum(objs.values()))

    def test_non_fluent_multi_sum(self):
        objs = {('x1', 'x1'): 1,
                ('x1', 'x2'): 2,
                ('x2', 'x1'): 3,
                ('x2', 'x2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        obj : object;
                    }};
                    pvariables {{ 
                        C(obj, obj) : {{ non-fluent, int, default = -1}};
                        p : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = sum_{{?x : obj, ?y : obj}}[C(?x, ?y)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    non-fluents {{
                        {'; '.join(f'C({",".join(o)}) = {v}' for o, v in objs.items())};
                    }};
                    objects {{
                        obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, sum(objs.values()))

    def test_non_fluent_multi_sum2(self):
        objs = {('x1', 'y1'): 1,
                ('x1', 'y2'): 2,
                ('x2', 'y1'): 3,
                ('x2', 'y2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_obj : object;
                        y_obj : object; 
                    }};
                    pvariables {{ 
                        C(x_obj, y_obj) : {{ non-fluent, int, default = -1}};
                        p : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = 2 * sum_{{?x : x_obj, ?y : y_obj}}[C(?x, ?y)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    non-fluents {{
                        {'; '.join(f'C({",".join(o)}) = {v}' for o, v in objs.items())};
                    }};
                    objects {{
                        x_obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                        y_obj : {{{', '.join({y for _, y in objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 2 * sum(objs.values()))

    def test_invalid_fluent_prod(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, int, default = -1 }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = prod_{{?x : obj}}[q(?x)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={v}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        with self.assertRaises(AssertionError):
            conv.convert_str(rddl)

    def test_non_fluent_prod(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        C(obj) : {{ non-fluent, int, default = -1}};
                        p : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = prod_{{?x : obj}}[C(?x)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    non-fluents {{
                        {'; '.join(f'C({o})={v}' for o, v in objs.items())};
                    }};
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, np.prod(list(objs.values())))

    def test_non_fluent_multi_prod(self):
        objs = {('x1', 'y1'): 1,
                ('x1', 'y2'): 2,
                ('x2', 'y1'): 3,
                ('x2', 'y2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_obj : object;
                        y_obj : object; 
                    }};
                    pvariables {{ 
                        C(x_obj, y_obj) : {{ non-fluent, int, default = -1}};
                        p : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = 3 + prod_{{?x : x_obj, ?y : y_obj}}[C(?x, ?y)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    non-fluents {{
                        {'; '.join(f'C({",".join(o)}) = {v}' for o, v in objs.items())};
                    }};
                    objects {{
                        x_obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                        y_obj : {{{', '.join({y for _, y in objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 3 + np.prod(list(objs.values())))

    def test_const_prod(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        C(obj) : {{ non-fluent, int, default = -1}};
                        p : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = prod_{{?x : obj}}[2]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    non-fluents {{
                        {'; '.join(f'C({o})={v}' for o, v in objs.items())};
                    }};
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, np.power(2, len(objs)))

    def test_fluent_forall_self(self):
        objs = {'x1': True, 'x2': True, 'x3': True, 'x4': True}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(obj) : {{ state-fluent, bool, default = false }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : obj}}[ p ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_fluent_forall_true(self):
        objs = {'x1': True, 'x2': True, 'x3': True, 'x4': True}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, bool, default = false }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : obj}}[ q(?x) ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, all(objs.values()))

    def test_fluent_forall_false(self):
        objs = {'x1': True, 'x2': False, 'x3': True, 'x4': True}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, bool, default = false }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : obj}}[ q(?x) ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, all(objs.values()))

    def test_fluent_forall_num(self):
        objs = {'x1': True, 'x2': True, 'x3': True, 'x4': True}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, bool, default = false }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : obj}}[ 0 * q(?x) ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)

    def test_const_forall_true(self):
        objs = {'x1': False, 'x2': False, 'x3': False, 'x4': False}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : obj}}[ true ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_const_forall_false(self):
        objs = {'x1': False, 'x2': False, 'x3': False, 'x4': False}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : obj}}[ false ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)

    def test_forall_multi_param(self):
        objs = {('x1', 'y1'): True, ('x1', 'y2'): True, ('x2', 'y1'): True, ('x2', 'y2'): True}
        rddl = f'''
                domain my_test {{
                    types {{ obj1 : object; obj2 : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj1, obj2) : {{ state-fluent, bool, default = false }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : obj1, ?y : obj2}}[ q(?x,?y) ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj1 : {{{', '.join(set(obj[0] for obj in objs.keys()))}}};
                        obj2 : {{{', '.join(set(obj[1] for obj in objs.keys()))}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o[0]},{o[1]})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, all(objs.values()))

    def test_forall_rel(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : obj}}[ q(?x) >= 1 ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={v}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertEqual(len(dyn.branch.planes), len(objs))
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, all(q >= 1 for q in objs.values()))

    def test_forall_or(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : obj}}[ ~ (~(q(?x) >= 1) ^ ~p)  ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={v}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, all(q >= 1 or p_ for q in objs.values()))

    def test_forall_forall(self):
        objs = {('x1', 'y1'): 1,
                ('x1', 'y2'): 2,
                ('x2', 'y1'): 3,
                ('x2', 'y2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_obj : object;
                        y_obj : object; 
                    }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(x_obj, y_obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : x_obj}}[ forall_{{?y : y_obj}}[ q(?x,?y) > 0 ]]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        x_obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                        y_obj : {{{', '.join({y for _, y in objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({",".join(o)}) = {v}' for o, v in objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertFalse(dyn.branch.isConjunction)
        self.assertEqual(len(dyn.branch.planes), len(objs))
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, all(all(objs[(x, y)] > 0
                                    for y in set(y for _, y in objs.keys()))
                                for x in set(x for x, _ in objs.keys())))

    def test_forall_exists(self):
        objs = {('x1', 'y1'): 1,
                ('x1', 'y2'): 3,
                ('x2', 'y1'): 2,
                ('x2', 'y2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_obj : object;
                        y_obj : object; 
                    }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(x_obj, y_obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : x_obj}}[ exists_{{?y : y_obj}}[ q(?x,?y) >= 3 ]]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        x_obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                        y_obj : {{{', '.join({y for _, y in objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({",".join(o)}) = {v}' for o, v in objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertFalse(dyn.branch.isConjunction)
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, all(any(objs[(x, y)] >= 3
                                    for y in set(y for _, y in objs.keys()))
                                for x in set(x for x, _ in objs.keys())))

    def test_forall_not_exists(self):
        objs = {('x1', 'y1'): 1,
                ('x1', 'y2'): 3,
                ('x2', 'y1'): 2,
                ('x2', 'y2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_obj : object;
                        y_obj : object; 
                    }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(x_obj, y_obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : x_obj}}[ ~ exists_{{?y : y_obj}}[ q(?x,?y) < 1 ]]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        x_obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                        y_obj : {{{', '.join({y for _, y in objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({",".join(o)}) = {v}' for o, v in objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertFalse(dyn.branch.isConjunction)
        self.assertEqual(len(dyn.branch.planes), len(objs))
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, all(not any(objs[(x, y)] < 1
                                        for y in set(y for _, y in objs.keys()))
                                for x in set(x for x, _ in objs.keys())))

    def test_forall_and(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : obj}}[ q(?x) >= 1 ^ p  ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={v}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertEqual(len(dyn.branch.planes), len(objs) * 2)  # forall AND x AND inside forall
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, all(q >= 1 and p_ for q in objs.values()))

    def test_invalid_forall_if(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = forall_{{?x : obj}}[ if (q(?x) > 2) then true else false ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={v}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        with self.assertRaises(ValueError):
            conv.convert_str(rddl)

    def test_exists_self(self):
        objs = {'x1': True, 'x2': True, 'x3': True, 'x4': True}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(obj) : {{ state-fluent, bool, default = false }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ p ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_exists_true(self):
        objs = {'x1': False, 'x2': False, 'x3': True, 'x4': False}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, bool, default = false }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ q(?x) ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, any(objs.values()))

    def test_exists_false(self):
        objs = {'x1': False, 'x2': False, 'x3': False, 'x4': False}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, bool, default = false }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ q(?x) ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, any(objs.values()))

    def test_exists_num(self):
        objs = {'x1': True, 'x2': True, 'x3': True, 'x4': True}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, bool, default = false }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ 0 * q(?x) ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)

    def test_const_exists_true(self):
        objs = {'x1': False, 'x2': False, 'x3': False, 'x4': False}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ true ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_const_exists_false(self):
        objs = {'x1': False, 'x2': False, 'x3': False, 'x4': False}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ false ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)

    def test_exists_rel(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ q(?x) > 3 ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={v}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertFalse(dyn.branch.isConjunction)
        self.assertEqual(len(dyn.branch.planes), len(objs))
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, any(q > 3 for q in objs.values()))

    def test_exists_multi_param(self):
        objs = {('x1', 'y1'): False, ('x2', 'y2'): False, ('x3', 'y3'): True, ('x4', 'y4'): False}
        rddl = f'''
                domain my_test {{
                    types {{ obj1 : object; obj2 : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj1, obj2) : {{ state-fluent, bool, default = false }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj1, ?y : obj2}}[ q(?x,?y) ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj1 : {{{', '.join([obj[0] for obj in objs.keys()])}}};
                        obj2 : {{{', '.join([obj[1] for obj in objs.keys()])}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o[0]},{o[1]})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, any(objs.values()))

    def test_exists_var_equals(self):
        objs = ['x1', 'x2', 'x3', 'x4']
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q : {{ state-fluent, obj, default = none }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ q == ?x]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs)}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{ q={objs[2]};}}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertEqual(len(dyn.branch.planes), len(objs))  # disjunction over all values
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(q, objs[2])
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_exists_one_var_equals(self):
        objs = ['x1']
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q : {{ state-fluent, obj, default = none }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ q == ?x]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs)}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{ q={objs[0]};}}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertNotIn(None, dyn.children)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(q, objs[0])
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_exists_var_equals_and(self):
        objs = ['x1', 'x2', 'x3', 'x4']
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q : {{ state-fluent, obj, default = none }};
                        r : {{ state-fluent, int, default = 2 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ ?x == q ^ r < 3]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs)}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{ q={objs[2]};}}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertEqual(len(dyn.branch.planes), 1)  # only one top comparison/branch
        self.assertEqual(len(dyn.branch.planes[0][1]), len(objs))  # switch with branch for each value
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(q, objs[2])
        r = conv.world.getState(WORLD, 'r', unique=True)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, r < 3)

    def test_exists_multi_param_var_equals(self):
        objs_x = ['x1', 'x2']
        objs_y = ['y1', 'y2', 'y3', 'y4']
        rddl = f'''
                domain my_test {{
                    types {{ obj_x : object; obj_y : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj_x) : {{ state-fluent, obj_y, default = none }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj_x, ?y : obj_y}}[ q(?x) == ?y ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj_x : {{{', '.join(objs_x)}}};
                        obj_y : {{{', '.join(objs_y)}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{ q({objs_x[1]})={objs_y[2]};}}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertEqual(len(dyn.branch.planes), len(objs_x) * len(objs_y))  # disjunction over all combinations
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        q = conv.world.getState(WORLD, conv.get_feature_name(('q', objs_x[1])), unique=True)
        self.assertEqual(q, objs_y[2])
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_exists_multi_param_var_equals_and(self):
        objs_x = ['x1', 'x2']
        objs_y = ['y1', 'y2', 'y3', 'y4']
        rddl = f'''
                domain my_test {{
                    types {{ obj_x : object; obj_y : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj_x) : {{ state-fluent, obj_y, default = none }};
                        r : {{ state-fluent, int, default = 2 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj_x, ?y : obj_y}}[ q(?x) == ?y ^ r < 3]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj_x : {{{', '.join(objs_x)}}};
                        obj_y : {{{', '.join(objs_y)}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{ q({objs_x[1]})={objs_y[2]};}}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertEqual(len(dyn.branch.planes), 1)
        self.assertEqual(len(dyn.branch.planes[0][1]), len(objs_y))  # switch over first x value
        self.assertEqual(len(dyn.children[None].branch.planes), 1)
        self.assertEqual(len(dyn.children[None].branch.planes[0][1]), len(objs_y))  # switch over second x value
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        q = conv.world.getState(WORLD, conv.get_feature_name(('q', objs_x[1])), unique=True)
        self.assertEqual(q, objs_y[2])
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, True)

    def test_exists_and(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ q(?x) > 3 ^ p  ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={v}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, any(q > 3 and p_ for q in objs.values()))

    def test_exists_or(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ q(?x) > 3 | p  ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={v}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertFalse(dyn.branch.isConjunction)
        self.assertEqual(len(dyn.branch.planes), len(objs) * 2)  # exists OR x OR inside exists
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, False)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, any(q > 3 or p_ for q in objs.values()))

    def test_exists_exists(self):
        objs = {('x1', 'y1'): 1,
                ('x1', 'y2'): 2,
                ('x2', 'y1'): 3,
                ('x2', 'y2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_obj : object;
                        y_obj : object; 
                    }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(x_obj, y_obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : x_obj}}[ exists_{{?y : y_obj}}[ q(?x,?y) > 3 ]]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        x_obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                        y_obj : {{{', '.join({y for _, y in objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({",".join(o)}) = {v}' for o, v in objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertFalse(dyn.branch.isConjunction)
        self.assertEqual(len(dyn.branch.planes), len(objs))
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, any(any(objs[(x, y)] > 3
                                    for y in set(y for _, y in objs.keys()))
                                for x in set(x for x, _ in objs.keys())))

    def test_exists_forall(self):
        objs = {('x1', 'y1'): 1,
                ('x1', 'y2'): 2,
                ('x2', 'y1'): 3,
                ('x2', 'y2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_obj : object;
                        y_obj : object; 
                    }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(x_obj, y_obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : x_obj}}[ forall_{{?y : y_obj}}[ q(?x,?y) >= 3 ]]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        x_obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                        y_obj : {{{', '.join({y for _, y in objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({",".join(o)}) = {v}' for o, v in objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertFalse(dyn.branch.isConjunction)
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, any(all(objs[(x, y)] >= 3
                                    for y in set(y for _, y in objs.keys()))
                                for x in set(x for x, _ in objs.keys())))

    def test_exists_not_forall(self):
        objs = {('x1', 'y1'): 1,
                ('x1', 'y2'): 2,
                ('x2', 'y1'): 3,
                ('x2', 'y2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_obj : object;
                        y_obj : object; 
                    }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = true }};
                        q(x_obj, y_obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : x_obj}}[ ~ forall_{{?y : y_obj}}[ q(?x,?y) >= 3 ]]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        x_obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                        y_obj : {{{', '.join({y for _, y in objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({",".join(o)}) = {v}' for o, v in objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        dyn = conv.world.getDynamics(stateKey(WORLD, 'p'), True)[0]
        self.assertFalse(dyn.branch.isConjunction)
        self.assertEqual(len(dyn.branch.planes), len(objs))
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, True)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, any(not all(objs[(x, y)] >= 3
                                        for y in set(y for _, y in objs.keys()))
                                for x in set(x for x, _ in objs.keys())))

    def test_invalid_exists_if(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, bool, default = false }};
                        q(obj) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = exists_{{?x : obj}}[ if (q(?x) > 2) then q(?x) else - q(?x) ]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={v}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        with self.assertRaises(ValueError):
            conv.convert_str(rddl)


if __name__ == '__main__':
    unittest.main()
