import unittest
from psychsim.pwl import WORLD, stateKey
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class TestDynamics(unittest.TestCase):

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

    def test_actions_condition(self):
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

    def test_actions_condition_or(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        a1 : { action-fluent, bool, default = false }; 
                        a2 : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if ( a1 | a2 ) then p + 1 else 2; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a1; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertIn(stateKey(WORLD, 'p'), conv.world.dynamics)
        self.assertEqual(len(conv.world.dynamics[stateKey(WORLD, 'p')]), 3)  # 2 actions + True
        self.assertIn(True, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[True]), 1)
        for action in next(iter(conv.world.agents.values())).actions:
            self.assertIn(action, conv.world.dynamics[stateKey(WORLD, 'p')])
            self.assertEqual(len(conv.world.dynamics[action]), 1)

    def test_action_condition_or_fluent(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        q : { state-fluent,  bool, default = false };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if ( a | q ) then p + 1 else 2; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertIn(stateKey(WORLD, 'p'), conv.world.dynamics)
        self.assertEqual(len(conv.world.dynamics[stateKey(WORLD, 'p')]), 2)  # 1 action + True
        self.assertIn(True, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[True]), 1)
        action = next(iter(next(iter(conv.world.agents.values())).actions))
        self.assertIn(action, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[action]), 1)

    def test_condition_or(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        q : { state-fluent,  bool, default = false };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if ( p < 4 | q ) then p + 1 else 2; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertIn(stateKey(WORLD, 'p'), conv.world.dynamics)
        self.assertEqual(len(conv.world.dynamics[stateKey(WORLD, 'p')]), 1)  # True
        self.assertIn(True, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[True]), 1)
        action = next(iter(next(iter(conv.world.agents.values())).actions))
        self.assertNotIn(action, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[action]), 0)

    def test_actions_condition_and(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        a1 : { action-fluent, bool, default = false }; 
                        a2 : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if ( a1 ^ a2 ) then p + 1 else 123; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a1; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertIn(stateKey(WORLD, 'p'), conv.world.dynamics)
        self.assertEqual(len(conv.world.dynamics[stateKey(WORLD, 'p')]), 1)  # True
        self.assertIn(True, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[True]), 1)
        for action in next(iter(conv.world.agents.values())).actions:
            self.assertIn(action, conv.world.dynamics)
            self.assertEqual(len(conv.world.dynamics[action]), 0)  # not conditioned
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 123)

    def test_action_condition_and_fluent(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        q : { state-fluent,  bool, default = false };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if ( a ^ q ) then p + 1 else 123; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertIn(stateKey(WORLD, 'p'), conv.world.dynamics)
        self.assertEqual(len(conv.world.dynamics[stateKey(WORLD, 'p')]), 2)  # action + world
        self.assertIn(True, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertIn(True, conv.world.dynamics)
        action = next(iter(next(iter(conv.world.agents.values())).actions))
        self.assertIn(action, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[action]), 1)
        p = conv.world.getState(WORLD, 'p', unique=True)
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(p, 1)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, p + 1 if q else 123)

    def test_actions_condition_or_and_fluent(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        q : { state-fluent,  bool, default = false };
                        r : { state-fluent,  bool, default = false };
                        a1 : { action-fluent, bool, default = false };
                        a2 : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if (( a1 ^ q ) | ( a2 ^ r )) then p + 1 else 2; }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertIn(stateKey(WORLD, 'p'), conv.world.dynamics)
        self.assertEqual(len(conv.world.dynamics[stateKey(WORLD, 'p')]), 3)  # 2 actions + True
        self.assertIn(True, conv.world.dynamics[stateKey(WORLD, 'p')])
        self.assertEqual(len(conv.world.dynamics[True]), 1)
        for action in next(iter(conv.world.agents.values())).actions:
            self.assertIn(action, conv.world.dynamics[stateKey(WORLD, 'p')])
            self.assertEqual(len(conv.world.dynamics[action]), 1)

    def test_actions_condition_exists(self):
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

    def test_actions_condition_exists_and(self):
        rddl = '''
                domain my_test {
                    types { agent : object; };
                    pvariables { 
                        p         : { state-fluent,  int, default = 0 };
                        q(agent)  : { state-fluent,  int, default = 0 };
                        a1(agent) : { action-fluent, bool, default = false }; 
                        a2(agent) : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = if ( exists_{?a : agent} [a1(?a) ^ q(?a) < 3] ) then
                                    p + 1
                                else if ( exists_{?a : agent} [a2(?a)] ^ q(?a) > -3) then
                                    p - 1
                                else
                                    0;
                    };
                    reward = 0;
                }
                non-fluents my_test_empty { 
                    domain = my_test; 
                    objects { agent: { Paul, John, George, Ringo }; };
                 }
                instance my_test_inst { domain = my_test; horizon  = 2; }
                '''
        conv = Converter(const_as_assert=True)
        conv.convert_str(rddl)
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

    def test_actions_param_condition(self):
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
