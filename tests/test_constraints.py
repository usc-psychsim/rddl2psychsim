import unittest
from psychsim.pwl import WORLD, actionKey, makeTree
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
                    state-action-constraints { ~a; };   // no legal actions
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        ag_name = next(iter(conv.world.agents.keys()))
        legal_acts = conv.world.agents[ag_name].getLegalActions()
        self.assertEqual(len(legal_acts), 0)

    def test_actions_legal_constraint(self):
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
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, 0)
        conv.world.step()
        ag_name = next(iter(conv.world.agents.keys()))
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(q, 1)
        a = conv.world.getFeature(actionKey(ag_name), unique=True)
        a1 = conv.actions[ag_name]['a1']
        a2 = conv.actions[ag_name]['a2']
        self.assertEqual(a, a1 if q > 1 else a2)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, p_ + 2 if a == a1 else p_ - 2)

    def test_actions_legal_precondition(self):
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
                    action-preconditions { a1 => q > 1; a2 => q <= 1; }; 
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a1; }; horizon  = 3; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, 0)
        conv.world.step()
        ag_name = next(iter(conv.world.agents.keys()))
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(q, 1)
        a = conv.world.getFeature(actionKey(ag_name), unique=True)
        a1 = conv.actions[ag_name]['a1']
        a2 = conv.actions[ag_name]['a2']
        self.assertEqual(a, a1 if q > 1 else a2)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, p_ + 2 if a == a1 else p_ - 2)

    def test_actions_legal_logic(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        q : { state-fluent,  bool, default = true };
                        r : { state-fluent,  bool, default = false };
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
                    state-action-constraints { a1 => q ^ r; a2 => q | r; }; 
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a1; }; horizon  = 3; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, 0)
        conv.world.step()
        ag_name = next(iter(conv.world.agents.keys()))
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(q, True)
        r = conv.world.getState(WORLD, 'r', unique=True)
        self.assertEqual(r, False)
        a = conv.world.getFeature(actionKey(ag_name), unique=True)
        a1 = conv.actions[ag_name]['a1']
        a2 = conv.actions[ag_name]['a2']
        self.assertEqual(a, a1 if q and r else a2)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, p_ + 2 if a == a1 else p_ - 2)

    def test_actions_legal_const(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        q : { non-fluent,  int, default = 1 };
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

        ag_name, agent = next(iter(conv.world.agents.items()))
        a1 = conv.actions[ag_name]['a1']
        a2 = conv.actions[ag_name]['a2']
        self.assertIn(a1, agent.legal)
        self.assertEqual(agent.legal[a1], makeTree(False))
        self.assertNotIn(a2, agent.legal)

        legal_acts = conv.world.agents[ag_name].getLegalActions()
        self.assertIn(a2, legal_acts)

        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, p_ - 2)

    def test_actions_param_legal(self):
        objs = {'x1': True, 'x2': False, 'x3': False, 'x4': True}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p(obj) : {{ state-fluent, bool, default = false }};
                        a(obj) : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p(?o)' = p(?o); }};
                    reward = 0;
                    state-action-constraints {{ forall_{{?o : obj}}[ a(?o) => p(?o) ]; }}; 
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{ obj : {{{', '.join(objs.keys())}}}; }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'p({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        conv.world.step()
        ag_name = next(iter(conv.world.agents.keys()))
        legal_acts = conv.world.agents[ag_name].getLegalActions()
        for name, val in objs.items():
            p = conv.world.getState(WORLD, Converter.get_feature_name(('p', name)), unique=True)
            self.assertEqual(p, val)
            a = conv.actions[ag_name][Converter.get_feature_name(('a', name))]
            if val:
                self.assertIn(a, legal_acts)
            else:
                self.assertNotIn(a, legal_acts)

    def test_actions_param_legal_ma(self):
        agents = {'John': 1.22, 'Paul': 3.75, 'George': -1.14, 'Ringo': 4.73}
        objs = {'x1': True, 'x2': False, 'x3': False, 'x4': True}
        rddl = f'''
                domain my_test {{
                    types {{ agent : object; obj: object; }};
                    pvariables {{ 
                        p(obj, agent) : {{ state-fluent, real, default = 0 }};
                        q(obj) : {{ state-fluent, bool, default = True }};
                        a(obj, agent) : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p'(?o, ?a) = p(?o, ?a); }};
                    reward = 0;
                    state-action-constraints {{ forall_{{?o: obj, ?a: agent}}[ a(?o, ?a) => q(?o) ]; }}; 
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{ 
                        agent : {{{', '.join(agents.keys())}}}; 
                        obj : {{{', '.join(objs.keys())}}};
                    }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{ 
                        {'; '.join(f'p({o},{a})={v if l else -1}' for a, v in agents.items() for o, l in objs.items())};
                        {'; '.join(f'q({o})={str(v).lower()}' for o, v in objs.items())};  
                    }}; 
                    horizon  = 0;
                }}
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        conv.world.step()
        for ag_name, val in agents.items():
            legal_acts = conv.world.agents[ag_name].getLegalActions()
            for obj_name, legal in objs.items():
                p = conv.world.getState(ag_name, Converter.get_feature_name(('p', obj_name)), unique=True)
                self.assertEqual(p, val if legal else -1)
                a = conv.actions[ag_name][Converter.get_feature_name(('a', obj_name, ag_name))]
                if legal:
                    self.assertIn(a, legal_acts)
                else:
                    self.assertNotIn(a, legal_acts)

    def test_actions_param_legal_logic(self):
        objs = {'x1': True, 'x2': False, 'x3': False, 'x4': True}
        rddl = f'''
                domain my_test {{
                    types {{ obj : object; }};
                    pvariables {{ 
                        p(obj) : {{ state-fluent, bool, default = false }};
                        a(obj) : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p(?o)' = p(?o); }};
                    reward = 0;
                    state-action-constraints {{ 
                        forall_{{?o : obj}}[ 
                            a(?o) => exists_{{?o2 : obj}} [ p(?o) ^ p(?o2) ] 
                        ]; 
                    }}; 
                }}
                non-fluents my_test_nf {{ 
                    domain = my_test; 
                    objects {{ obj : {{{', '.join(objs.keys())}}}; }};
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{
                        {'; '.join(f'p({o})={str(v).lower()}' for o, v in objs.items())}; 
                    }}; 
                }}
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        conv.world.step()
        ag_name = next(iter(conv.world.agents.keys()))
        legal_acts = conv.world.agents[ag_name].getLegalActions()
        for name, val in objs.items():
            p = conv.world.getState(WORLD, Converter.get_feature_name(('p', name)), unique=True)
            self.assertEqual(p, val)
            a = conv.actions[ag_name][Converter.get_feature_name(('a', name))]
            if val:
                self.assertIn(a, legal_acts)
            else:
                self.assertNotIn(a, legal_acts)


if __name__ == '__main__':
    unittest.main()
