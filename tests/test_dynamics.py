import unittest
from psychsim.pwl import WORLD, stateKey
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class TestMultiagent(unittest.TestCase):

    def test_fluent_const(self):
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
        conv.convert_str(rddl)
        self.assertIn(stateKey(WORLD, 'p'), conv.world.dynamics)
        self.assertIn(True, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[True]), 1)
        action = next(iter(next(iter(conv.world.agents.values())).actions))
        self.assertEqual(len(conv.world.dynamics[action]), 0)

    def test_fluent_other(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        q : { state-fluent,  int, default = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = 2; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertNotIn(stateKey(WORLD, 'q'), conv.world.dynamics)
        action = next(iter(next(iter(conv.world.agents.values())).actions))
        self.assertEqual(len(conv.world.dynamics[action]), 0)

    def test_fluent_self(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = p; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertNotIn(stateKey(WORLD, 'p'), conv.world.dynamics)
        self.assertNotIn(True, conv.world.dynamics)
        action = next(iter(next(iter(conv.world.agents.values())).actions))
        self.assertEqual(len(conv.world.dynamics[action]), 0)

    def test_fluent_inc_self(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = p + 1; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertIn(stateKey(WORLD, 'p'), conv.world.dynamics)
        self.assertIn(True, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[True]), 1)
        action = next(iter(next(iter(conv.world.agents.values())).actions))
        self.assertEqual(len(conv.world.dynamics[action]), 0)

    def test_action_condition(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (a) then p + 1 else 2; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertIn(stateKey(WORLD, 'p'), conv.world.dynamics)
        self.assertIn(True, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[True]), 1)
        action = next(iter(next(iter(conv.world.agents.values())).actions))
        self.assertIn(action, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[action]), 1)

    def test_action_condition_self(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (a) then p + 1 else p; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertIn(stateKey(WORLD, 'p'), conv.world.dynamics)
        self.assertNotIn(True, conv.world.dynamics[stateKey(WORLD, 'p')])
        action = next(iter(next(iter(conv.world.agents.values())).actions))
        self.assertIn(action, conv.world.dynamics)
        self.assertEqual(len(conv.world.dynamics[action]), 1)
        self.assertIn(action, conv.world.dynamics[stateKey(WORLD, 'p')])

    def test_actions_conditions(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        a1 : { action-fluent, bool, default = false }; 
                        a2 : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (a1) then
                                    p + 1
                                else if (a2) then
                                    p - 1
                                else
                                    0;
                    };
                    reward = -p;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a1; }; horizon  = 2; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        conv.world.step()
        for ag_name in conv.world.agents.keys():
            a1 = conv.actions[ag_name]['a1']
            a2 = conv.actions[ag_name]['a2']
            p = stateKey(WORLD, 'p')
            self.assertIn(a1, conv.world.dynamics)
            self.assertIn(a2, conv.world.dynamics)
            self.assertIn(True, conv.world.dynamics)
            self.assertIn(p, conv.world.dynamics[a1])
            self.assertIn(p, conv.world.dynamics[a2])
            self.assertIn(p, conv.world.dynamics[True])
            self.assertIn(a1, conv.world.dynamics[p])
            self.assertIn(a2, conv.world.dynamics[p])
            self.assertIn(True, conv.world.dynamics[p])

    def test_actions_conditions_multi(self):
        rddl = '''
                domain my_test {
                    types { agent : object; };
                    pvariables { 
                        p : { state-fluent,  int, default = 0 };
                        a1(agent) : { action-fluent, bool, default = false }; 
                        a2(agent) : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (exists_{?a : agent} [a1(?a)] ) then
                                    p + 1
                                else if ( exists_{?a : agent} [a2(?a)] ) then
                                    p - 1
                                else
                                    0;
                    };
                    reward = -p;
                }
                non-fluents my_test_empty { 
                    domain = my_test; 
                    objects { agent: { Paul, John, George, Ringo }; };
                 }
                instance my_test_inst { domain = my_test; init-state { p = 0; }; horizon  = 2; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        conv.world.step()
        for ag_name in conv.world.agents.keys():
            a1 = conv.actions[ag_name][Converter.get_feature_name(('a1', ag_name))]
            a2 = conv.actions[ag_name][Converter.get_feature_name(('a2', ag_name))]
            p = stateKey(WORLD, 'p')
            self.assertIn(a1, conv.world.dynamics)
            self.assertIn(a2, conv.world.dynamics)
            self.assertIn(True, conv.world.dynamics)
            self.assertIn(p, conv.world.dynamics[a1])
            self.assertIn(p, conv.world.dynamics[a2])
            self.assertIn(p, conv.world.dynamics[True])
            self.assertIn(a1, conv.world.dynamics[p])
            self.assertIn(a2, conv.world.dynamics[p])
            self.assertIn(True, conv.world.dynamics[p])

    def test_actions_param_conditions(self):
        agents = {'John': 1.22, 'Paul': 3.75, 'George': -1.14, 'Ringo': 4.73}
        rddl = f'''
                domain my_test {{
                    types {{ agent : object; }};
                    pvariables {{ 
                        p(agent) : {{ state-fluent,  real, default = 0 }};
                        a1(agent) : {{ action-fluent, bool, default = false }}; 
                        a2(agent) : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ p'(?a) = if ( a1(?a) ) then
                                    p(?a) + 1
                                else if ( a2(?a) ) then
                                    p(?a) - 1
                                else
                                    0;
                    }};
                    reward = - sum_{{?a : agent}} p(?a);
                }}
                non-fluents my_test_empty {{ 
                    domain = my_test;
                    objects {{ agent : {{ {", ".join(agents.keys())} }}; }}; 
                }}
                instance my_test_inst {{ 
                    domain = my_test; 
                    init-state {{ {'; '.join(f'p({a}) = {v}' for a, v in agents.items())}; }};
                    horizon = 0;
                }}
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
        conv.world.step()
        for ag_name in conv.world.agents.keys():
            a1 = conv.actions[ag_name][Converter.get_feature_name(('a1', ag_name))]
            a2 = conv.actions[ag_name][Converter.get_feature_name(('a2', ag_name))]
            p = stateKey(ag_name, 'p')
            self.assertIn(a1, conv.world.dynamics)
            self.assertIn(a2, conv.world.dynamics)
            self.assertIn(True, conv.world.dynamics)
            self.assertIn(p, conv.world.dynamics[a1])
            self.assertIn(p, conv.world.dynamics[a2])
            self.assertIn(p, conv.world.dynamics[True])
            self.assertIn(a1, conv.world.dynamics[p])
            self.assertIn(a2, conv.world.dynamics[p])
            self.assertIn(True, conv.world.dynamics[p])


if __name__ == '__main__':
    unittest.main()
