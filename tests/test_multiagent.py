import unittest
from collections import OrderedDict
from psychsim.pwl import WORLD, turnKey
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class TestMultiagent(unittest.TestCase):

    def test_single_agent(self):
        rddl = '''
            domain my_test {
                pvariables { a : { action-fluent, bool, default = false }; };
                cpfs { }; 
                reward = 0;
            }
            non-fluents my_test_empty { domain = my_test; }
            instance my_test_inst { domain = my_test; init-state { a; }; }
            '''
        conv = Converter()
        conv.convert_str(rddl)
        self.assertEqual(len(conv.world.agents), 1)

    def test_agents_names(self):
        agents = ['John', 'Paul', 'George', 'Ringo']
        rddl = f'''
            domain my_test {{
                types {{ agent : object; }};
                pvariables {{ 
                    a : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ 
                    agent : {{ {",".join(agents)} }};
                }}; 
            }}
            instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for ag_name in agents:
            self.assertIn(ag_name, conv.world.agents)

    def test_default_action(self):
        agents = ['John', 'Paul', 'George', 'Ringo']
        rddl = f'''
            domain my_test {{
                types {{ agent : object; }};
                pvariables {{ 
                    a : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ 
                    agent : {{ {",".join(agents)} }};
                }}; 
            }}
            instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for ag_name in agents:
            actions = set(map(str, conv.world.agents[ag_name].actions))
            self.assertEqual(len(actions), 1)
            self.assertIn(f'{ag_name}-a', actions)

    def test_action(self):
        agents = ['John', 'Paul', 'George', 'Ringo']
        rddl = f'''
            domain my_test {{
                types {{ agent : object; }};
                pvariables {{ 
                    p: {{ state-fluent, int, default = 0 }};
                    a(agent) : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ 
                    agent : {{ {",".join(agents)} }};
                }}; 
            }}
            instance my_test_inst {{ domain = my_test; init-state {{ p = 1; }}; }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for ag_name in agents:
            actions = set(map(str, conv.world.agents[ag_name].actions))
            self.assertEqual(len(actions), 1)
            self.assertIn(f'{ag_name}-a', actions)

    def test_action_multi_param(self):
        agents = ['John', 'Paul', 'George', 'Ringo']
        rddl = f'''
            domain my_test {{
                types {{ agent : object; obj: object; }};
                pvariables {{ 
                    p: {{ state-fluent, int, default = 0 }};
                    a(obj, agent) : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ 
                    agent : {{ {",".join(agents)} }};
                    obj: {{ o1 }};
                }}; 
            }}
            instance my_test_inst {{ domain = my_test; init-state {{ p = 1; }}; }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for ag_name in agents:
            actions = set(map(str, conv.world.agents[ag_name].actions))
            self.assertEqual(len(actions), 1)
            self.assertIn(f'{ag_name}-{Converter.get_feature_name(("a", "o1"))}', actions)

    def test_action_multi_param2(self):
        agents = ['John', 'Paul', 'George', 'Ringo']
        rddl = f'''
            domain my_test {{
                types {{ agent : object; obj: object; }};
                pvariables {{ 
                    p: {{ state-fluent, int, default = 0 }};
                    a(agent, obj) : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ 
                    agent : {{ {",".join(agents)} }};
                    obj: {{ o1 }};
                }}; 
            }}
            instance my_test_inst {{ domain = my_test; init-state {{ p = 1; }}; }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for ag_name in agents:
            actions = set(map(str, conv.world.agents[ag_name].actions))
            self.assertEqual(len(actions), 1)
            self.assertIn(f'{ag_name}-{Converter.get_feature_name(("a", "o1"))}', actions)

    def test_non_agent_fluent(self):
        agents = ['John', 'Paul', 'George', 'Ringo']
        rddl = f'''
            domain my_test {{
                types {{ agent : object; }};
                pvariables {{ 
                    p : {{ state-fluent, int, default = 0 }};
                    a : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ p' = p + 1; }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ 
                    agent : {{ {",".join(agents)} }};
                }}; 
            }}
            instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1)

    def test_non_fluent(self):
        agents = {'John': 1.22, 'Paul': 3.75, 'George': -1.14, 'Ringo': 4.73}
        rddl = f'''
            domain my_test {{
                types {{ agent : object; }};
                pvariables {{ 
                    C(agent) : {{ non-fluent, real, default = 1}};
                    a : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{  }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                non-fluents {{
                    {'; '.join(f'C({a}) = {v}' for a, v in agents.items())};
                }};
                objects {{ 
                    agent : {{ {", ".join(agents.keys())} }};
                }}; 
            }}
            instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for a, v in agents.items():
            self.assertIn(Converter.get_feature_name(('C', a)), conv.constants)
            self.assertEqual(conv.constants[Converter.get_feature_name(('C', a))], v)

    def test_fluent(self):
        agents = {'John': 1.22, 'Paul': 3.75, 'George': -1.14, 'Ringo': 4.73}
        rddl = f'''
            domain my_test {{
                types {{ agent : object; }};
                pvariables {{ 
                    p(agent) : {{ state-fluent, real, default = 0 }};
                    a : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ p'(?a) = p(?a) + 1; }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ agent : {{ {", ".join(agents.keys())} }}; }}; 
            }}
            instance my_test_inst {{ 
                domain = my_test; 
                init-state {{ {'; '.join(f'p({a}) = {v}' for a, v in agents.items())}; }}; 
            }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for a, v in agents.items():
            p = conv.world.getState(a, 'p', unique=True)
            self.assertEqual(p, v)
        conv.world.step()
        for a, v in agents.items():
            p = conv.world.getState(a, 'p', unique=True)
            self.assertEqual(p, v + 1)

    def test_fluent_multi_param(self):
        agents = {'John': 1.22, 'Paul': 3.75, 'George': -1.14, 'Ringo': 4.73}
        rddl = f'''
            domain my_test {{
                types {{ agent : object; obj: object; }};
                pvariables {{ 
                    p(obj, agent) : {{ state-fluent, real, default = 0 }};
                    a : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ p'(?o, ?a) = p(?o, ?a) + 1; }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ 
                    agent : {{ {", ".join(agents.keys())} }};
                    obj: {{ o1 }};
                }}; 
            }}
            instance my_test_inst {{ 
                domain = my_test; 
                init-state {{ {'; '.join(f'p(o1, {a}) = {v}' for a, v in agents.items())}; }}; 
            }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for a, v in agents.items():
            p = conv.world.getState(a, Converter.get_feature_name(('p', 'o1')), unique=True)
            self.assertEqual(p, v)
        conv.world.step()
        for a, v in agents.items():
            p = conv.world.getState(a, Converter.get_feature_name(('p', 'o1')), unique=True)
            self.assertEqual(p, v + 1)

    def test_fluent_multi_param2(self):
        agents = {'John': 1.22, 'Paul': 3.75, 'George': -1.14, 'Ringo': 4.73}
        rddl = f'''
            domain my_test {{
                types {{ agent : object; obj: object; }};
                pvariables {{ 
                    p(agent, obj) : {{ state-fluent, real, default = 0 }};
                    a : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ p'(?a, ?o) = p(?a, ?o) + 1; }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ 
                    agent : {{ {", ".join(agents.keys())} }};
                    obj: {{ o1 }};
                }}; 
            }}
            instance my_test_inst {{ 
                domain = my_test; 
                init-state {{ {'; '.join(f'p({a}, o1) = {v}' for a, v in agents.items())}; }}; 
            }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for a, v in agents.items():
            p = conv.world.getState(a, Converter.get_feature_name(('p', 'o1')), unique=True)
            self.assertEqual(p, v)
        conv.world.step()
        for a, v in agents.items():
            p = conv.world.getState(a, Converter.get_feature_name(('p', 'o1')), unique=True)
            self.assertEqual(p, v + 1)

    def test_all_concurrent(self):
        agents = OrderedDict({'John': 1.22, 'Paul': 3.75, 'George': -1.14, 'Ringo': 4.73})
        rddl = f'''
            domain my_test {{
                requirements {{ concurrent }};
                types {{ agent : object; }};
                pvariables {{ 
                    p(agent) : {{ state-fluent, real, default = 0 }};
                    q : {{ observ-fluent, real }};     // observable to condition on agents' p
                    act : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ 
                    p'(?a) = p(?a) + 1;
                    q' = sum_{{?a : agent}}[ p(?a) ];   // sum current agents' p vars
                }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ agent : {{ {", ".join(agents.keys())} }}; }}; 
            }}
            instance my_test_inst {{ 
                domain = my_test; 
                init-state {{ {'; '.join(f'p({a}) = {v}' for a, v in agents.items())}; }}; 
            }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for a, v in agents.items():
            p = conv.world.getState(a, '__p', unique=True)
            self.assertEqual(p, v)
            turn = conv.world.getFeature(turnKey(a), unique=True)
            self.assertEqual(turn, 0)  # all agents on same turn
        conv.world.step()
        for a, v in agents.items():
            p = conv.world.getState(a, '__p', unique=True)
            self.assertEqual(p, v + 1)
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(q, sum(v + 1 for v in agents.values()))

    def test_all_sequential(self):
        agents = ['John', 'Paul', 'George', 'Ringo']
        rddl = f'''
            domain my_test {{
                types {{ agent : object; }};
                pvariables {{ 
                    p(agent) : {{ state-fluent, real, default = 0 }};
                    q : {{ observ-fluent, real }};     // observable to condition on agents' p
                    act(agent) : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ 
                    p'(?a) = if( act(?a) ) then
                                p(?a) + 1   // action-conditioned dynamics
                            else
                                p(?a);

                    q' = sum_{{?a : agent}}[ p(?a) ]; // sum current agents' p vars
                }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ agent : {{ {", ".join(agents)} }}; }}; 
            }}
            instance my_test_inst {{ domain = my_test; init-state {{ q = 0; }}; }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for i, a in enumerate(agents):
            p = conv.world.getState(a, '__p', unique=True)
            self.assertEqual(p, 0)
            turn = conv.world.getFeature(turnKey(a), unique=True)
            self.assertEqual(turn, i)
        q_sum = 0.
        for a in agents:
            conv.world.step()
            p = conv.world.getState(a, '__p', unique=True)
            self.assertEqual(p, 1)
            q_sum += 1
            q = conv.world.getState(WORLD, 'q', unique=True)
            self.assertEqual(q, q_sum)

    def test_partial_concurrent(self):
        agents = ['John', 'Paul', 'George', 'Ringo']
        rddl = f'''
            domain my_test {{
                requirements {{ concurrent }};
                types {{ agent : object; }};
                pvariables {{ 
                    p(agent) : {{ state-fluent, real, default = 0 }};
                    q : {{ observ-fluent, real }};     // observable to condition on agents' p
                    act(agent) : {{ action-fluent, bool, default = false }}; 
                }};
                cpfs {{ 
                    p'(?a) = if( act(?a) ) then
                                p(?a) + 1   // action-conditioned dynamics
                            else
                                p(?a);

                    q' = sum_{{?a : agent}}[ p(?a) ]; // sum current agents' p vars
                }};
                reward = 0;
            }}
            non-fluents my_test_empty {{ 
                domain = my_test;
                objects {{ agent : {{ {", ".join(agents)} }}; }}; 
            }}
            instance my_test_inst {{ 
                domain = my_test; 
                init-state {{ q = 0; }};
                max-nondef-actions = {len(agents) - 1}; // one agent acts after the others 
            }}
            '''
        conv = Converter()
        conv.convert_str(rddl)
        for i, a in enumerate(agents):
            p = conv.world.getState(a, '__p', unique=True)
            self.assertEqual(p, 0)
            turn = conv.world.getFeature(turnKey(a), unique=True)
            self.assertEqual(turn, 0 if i < len(agents) - 1 else 1)
        conv.world.step()
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(q, len(agents) - 1)
        conv.world.step()
        q = conv.world.getState(WORLD, 'q', unique=True)
        self.assertEqual(q, len(agents))


if __name__ == '__main__':
    unittest.main()
