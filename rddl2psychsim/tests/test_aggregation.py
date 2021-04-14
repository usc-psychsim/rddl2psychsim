import unittest
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

AG_NAME = 'Agent'


class TestAggregation(unittest.TestCase):

    def test_fluent_param_sum(self):
        pos_objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ pos : object; }};
                    pvariables {{ 
                        p : {{ state-fluent, int, default = -1 }};
                        q(pos) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = sum_{{?x : pos}}[q(?x)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        pos : {{{', '.join(pos_objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({o})={v}' for o, v in pos_objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, sum(pos_objs.values()))

    def test_fluent_multi_param_sum(self):
        pos_objs = {('x1', 'x1'): 1,
                    ('x1', 'x2'): 2,
                    ('x2', 'x1'): 3,
                    ('x2', 'x2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        pos : object;
                    }};
                    pvariables {{ 
                        p : {{ state-fluent, int, default = -1 }};
                        q(pos, pos) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = sum_{{?x : pos, ?y : pos}}[q(?x, ?y)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        pos : {{{', '.join({x for x, _ in pos_objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({",".join(o)}) = {v}' for o, v in pos_objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, sum(pos_objs.values()))

    def test_fluent_multi_param_sum2(self):
        pos_objs = {('x1', 'y1'): 1,
                    ('x1', 'y2'): 2,
                    ('x2', 'y1'): 3,
                    ('x2', 'y2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_pos : object;
                        y_pos : object; 
                    }};
                    pvariables {{ 
                        p : {{ state-fluent, int, default = -1 }};
                        q(x_pos, y_pos) : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = sum_{{?x : x_pos, ?y : y_pos}}[q(?x, ?y)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        x_pos : {{{', '.join({x for x, _ in pos_objs.keys()})}}};
                        y_pos : {{{', '.join({y for _, y in pos_objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'q({",".join(o)}) = {v}' for o, v in pos_objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, sum(pos_objs.values()))

    def test_non_fluent_param_sum(self):
        pos_objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ pos : object; }};
                    pvariables {{ 
                        C(pos) : {{ non-fluent, int, default = -1}};
                        p : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = sum_{{?x : pos}}[C(?x)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    non-fluents {{
                        {'; '.join(f'C({o})={v}' for o, v in pos_objs.items())};
                    }};
                    objects {{
                        pos : {{{', '.join(pos_objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, sum(pos_objs.values()))

    def test_non_fluent_multi_param_sum(self):
        pos_objs = {('x1', 'x1'): 1,
                    ('x1', 'x2'): 2,
                    ('x2', 'x1'): 3,
                    ('x2', 'x2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        pos : object;
                    }};
                    pvariables {{ 
                        C(pos, pos) : {{ non-fluent, int, default = -1}};
                        p : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = sum_{{?x : pos, ?y : pos}}[C(?x, ?y)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    non-fluents {{
                        {'; '.join(f'C({",".join(o)}) = {v}' for o, v in pos_objs.items())};
                    }};
                    objects {{
                        pos : {{{', '.join({x for x, _ in pos_objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, sum(pos_objs.values()))

    def test_non_fluent_multi_param_sum2(self):
        pos_objs = {('x1', 'y1'): 1,
                    ('x1', 'y2'): 2,
                    ('x2', 'y1'): 3,
                    ('x2', 'y2'): 4}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_pos : object;
                        y_pos : object; 
                    }};
                    pvariables {{ 
                        C(x_pos, y_pos) : {{ non-fluent, int, default = -1}};
                        p : {{ state-fluent, int, default = -1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = sum_{{?x : x_pos, ?y : y_pos}}[C(?x, ?y)]; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    non-fluents {{
                        {'; '.join(f'C({",".join(o)}) = {v}' for o, v in pos_objs.items())};
                    }};
                    objects {{
                        x_pos : {{{', '.join({x for x, _ in pos_objs.keys()})}}};
                        y_pos : {{{', '.join({y for _, y in pos_objs.keys()})}}};
                    }};
                }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, -1)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, sum(pos_objs.values()))
