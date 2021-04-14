import unittest

from psychsim.pwl import stateKey
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

AG_NAME = 'Agent'


class TestTypes(unittest.TestCase):

    def test_int_fluent_def(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = 2; }; 
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

    def test_int_fluent_init(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = 2; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state {  p = 3; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 3)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 2)

    def test_bool_fluent_def(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  bool, default = true };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = false; }; 
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
        self.assertEqual(p, False)

    def test_bool_fluent_init(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  bool, default = true };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = true; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state {  p = false; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, False)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, True)

    def test_real_fluent_def(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  real, default = 3.14 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = -0.02; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 3.14)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, -.02)

    def test_real_fluent_init(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  real, default = 3.14 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = -0.01; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state {  p = 6.28; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 6.28)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, -0.01)

    def test_enum_fluent_def(self):
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

    def test_enum_fluent_init(self):
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

    def test_interm_fluent(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { interm-fluent, int, level = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = 2; }; 
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
        self.assertEqual(p, 2)

    def test_observ_fluent(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { observ-fluent, int };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = 2; }; 
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
        self.assertEqual(p, 2)

    def test_non_fluent_def(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        C : { non-fluent, int, default = 2};
                        p : { state-fluent, int, default = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = p + C; }; 
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
        self.assertEqual(p, 3)

    def test_non_fluent_init(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        C : { non-fluent, int, default = 2};
                        p : { state-fluent, int, default = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = p + C; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; non-fluents { C = -1; }; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 0)

    def test_fluent_param_def(self):
        objs = {'x1': True, 'x2': False, 'x3': False, 'x4': True}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p(obj) : {{ state-fluent,  bool, default = true }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p'(obj) = false; }}; 
                    reward = 0;
                }}
                non-fluents my_test_empty {{ 
                    domain = my_test;
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }}; 
                }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        for o, v in objs.items():
            p = conv.world.getState(AG_NAME, Converter.get_fluent_name(('p', o)), unique=True)
            self.assertEqual(p, True)
        conv.world.step()
        for o, v in objs.items():
            p = conv.world.getState(AG_NAME, Converter.get_fluent_name(('p', o)), unique=True)
            self.assertEqual(p, False)

    def test_fluent_param_init(self):
        objs = {'x1': True, 'x2': False, 'x3': False, 'x4': True}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p(obj) : {{ state-fluent,  bool, default = true }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p'(obj) = false; }}; 
                    reward = 0;
                }}
                non-fluents my_test_empty {{ 
                    domain = my_test;
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }}; 
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'p({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        for o, v in objs.items():
            p = conv.world.getState(AG_NAME, Converter.get_fluent_name(('p', o)), unique=True)
            self.assertEqual(p, v)
        conv.world.step()
        for o, v in objs.items():
            p = conv.world.getState(AG_NAME, Converter.get_fluent_name(('p', o)), unique=True)
            self.assertEqual(p, False)

    def test_fluent_multi_param_init(self):
        objs = {('x1', 'x1'): True,
                ('x1', 'x2'): False,
                ('x2', 'x1'): False,
                ('x2', 'x2'): True}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p(obj, obj) : {{ state-fluent,  bool, default = true }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p'(obj, obj) = false; }}; 
                    reward = 0;
                }}
                non-fluents my_test_empty {{ 
                    domain = my_test;
                    objects {{
                        obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                    }}; 
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'p({",".join(o)}) = {str(v).lower()}' for o, v in objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        for o, v in objs.items():
            p = conv.world.getState(AG_NAME, Converter.get_fluent_name(('p', *o)), unique=True)
            self.assertEqual(p, v)
        conv.world.step()
        for o, v in objs.items():
            p = conv.world.getState(AG_NAME, Converter.get_fluent_name(('p', *o)), unique=True)
            self.assertEqual(p, False)

    def test_fluent_multi_param_init2(self):
        objs = {('x1', 'x1'): True,
                ('x1', 'x2'): False,
                ('x2', 'x1'): False,
                ('x2', 'x2'): True}
        rddl = f'''
                domain my_test {{
                    types {{ 
                        x_obj : object;
                        y_obj : object; 
                    }};
                    pvariables {{ 
                        p(x_obj, y_obj) : {{ state-fluent,  bool, default = true }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p'(x_obj, y_obj) = false; }}; 
                    reward = 0;
                }}
                non-fluents my_test_empty {{ 
                    domain = my_test;
                    objects {{
                        x_obj : {{{', '.join({x for x, _ in objs.keys()})}}};
                        y_obj : {{{', '.join({y for _, y in objs.keys()})}}};
                    }}; 
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'p({",".join(o)}) = {str(v).lower()}' for o, v in objs.items())};
                    }}; 
                }}
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        for o, v in objs.items():
            p = conv.world.getState(AG_NAME, Converter.get_fluent_name(('p', *o)), unique=True)
            self.assertEqual(p, v)
        conv.world.step()
        for o, v in objs.items():
            p = conv.world.getState(AG_NAME, Converter.get_fluent_name(('p', *o)), unique=True)
            self.assertEqual(p, False)

    def test_non_fluent_param_def(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        C(obj) : {{ non-fluent, int, default = -1}};
                        p : {{ state-fluent, int, default = 1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = p + 1; }}; 
                    reward = 0;
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        for o, v in objs.items():
            self.assertIn(Converter.get_fluent_name(('C', o)), conv.constants)
            self.assertEqual(conv.constants[Converter.get_fluent_name(('C', o))], -1)

    def test_non_fluent_param_init(self):
        objs = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        C(obj) : {{ non-fluent, int, default = -1}};
                        p : {{ state-fluent, int, default = 1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = p + 1; }}; 
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
        conv.convert_str(rddl, AG_NAME)
        for o, v in objs.items():
            self.assertIn(Converter.get_fluent_name(('C', o)), conv.constants)
            self.assertEqual(conv.constants[Converter.get_fluent_name(('C', o))], v)

    def test_non_fluent_multi_param_init(self):
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
                        p : {{ state-fluent, int, default = 1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = p + 1; }}; 
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
        conv.convert_str(rddl, AG_NAME)
        for o, v in objs.items():
            self.assertIn(Converter.get_fluent_name(('C', *o)), conv.constants)
            self.assertEqual(conv.constants[Converter.get_fluent_name(('C', *o))], v)

    def test_non_fluent_multi_param_init2(self):
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
                        p : {{ state-fluent, int, default = 1 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p' = p + 1; }}; 
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
        conv.convert_str(rddl, AG_NAME)
        for o, v in objs.items():
            self.assertIn(Converter.get_fluent_name(('C', *o)), conv.constants)
            self.assertEqual(conv.constants[Converter.get_fluent_name(('C', *o))], v)

    def test_partial_observability(self):
        rddl = '''
                domain my_test {
                    requirements = { partially-observed };
                    pvariables { 
                        p : { state-fluent, int, default = 1 };
                        q : { observ-fluent, int };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { q' = 3; p' = q + 1; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl, AG_NAME)
        agent = conv.world.agents[AG_NAME]
        self.assertNotIn(stateKey(AG_NAME, 'p'), agent.omega)
        self.assertIn(stateKey(AG_NAME, 'q'), agent.omega)

        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        conv.world.step()
        p = conv.world.getState(AG_NAME, 'p', unique=True)
        self.assertEqual(p, 4)
        q = conv.world.getState(AG_NAME, 'q', unique=True)
        self.assertEqual(q, 3)
