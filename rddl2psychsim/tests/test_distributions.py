import unittest
from psychsim.pwl import WORLD
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class TestTypes(unittest.TestCase):

    def test_kron_delta(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  int, default = 1 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = KronDelta(2); }; 
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
        self.assertEqual(p, 2)

    def test_dirac_delta(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  real, default = 1.4 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = DiracDelta(2.5); }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 1.4)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 2.5)

    def test_bernoulli(self):
        rddl = '''
                domain my_test {
                    pvariables { 
                        p : { state-fluent,  real, default = 0 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { p' = Bernoulli(.3); }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p')
        self.assertEqual(p[0], 0.7)
        self.assertEqual(p[1], 0.3)

    def test_discrete(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high}; 
                    };
                    pvariables { 
                        p : { state-fluent,  enum_level, default = @low };
                        q : { state-fluent,  int, default = 3 };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { 
                        p' = if (q >= 2) then
                                Discrete(enum_level, @low : 0.5, @medium : 0.2, @high : 0.3)
                            else
                                Discrete(enum_level, @low : 0.2, @medium : 0.5, @high : 0.3);
                    }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 'low')
        conv.world.step()
        p = conv.world.getState(WORLD, 'p')
        self.assertEqual(p['low'], 0.5)
        self.assertEqual(p['medium'], 0.2)
        self.assertEqual(p['high'], 0.3)

    def test_discrete_enum_invalid_non_const(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high}; 
                    };
                    pvariables { 
                        p : { state-fluent,  enum_level, default = @low };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { 
                        p' = Discrete(enum_level,
                                @low : if (p >= 2) then 0.5 else 0.2,
                                @medium : if (p >= 2) then 0.2 else 0.5,
                                @high : 0.3
                            );
                    }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        with self.assertRaises(AssertionError):
            conv.convert_str(rddl)

    def test_discrete_enum_const(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high}; 
                    };
                    pvariables { 
                        C : { non-fluent, int, default = -0.1 };
                        p : { state-fluent, enum_level, default = @low };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { 
                        p' = Discrete(enum_level, @low : 0.2 + C, @medium : 0.5 - C, @high : 0.3);
                    }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 'low')
        conv.world.step()
        p = conv.world.getState(WORLD, 'p')
        self.assertEqual(p['low'], 0.1)
        self.assertEqual(p['medium'], 0.6)
        self.assertEqual(p['high'], 0.3)

    def test_discrete_enum_invalid_sum(self):
        rddl = '''
                domain my_test {
                    types {
                        enum_level : {@low, @medium, @high}; 
                    };
                    pvariables { 
                        p : { state-fluent, enum_level, default = @low };
                        a : { action-fluent, bool, default = false }; 
                    };
                    cpfs { 
                        p' = Discrete(enum_level, @low : 0.2, @medium : 0.5);
                    }; 
                    reward = 0;
                }
                non-fluents my_test_empty { domain = my_test; }
                instance my_test_inst { domain = my_test; init-state { a; }; }
                '''
        conv = Converter()
        with self.assertRaises(AssertionError):
            conv.convert_str(rddl)

    def test_normal_const(self):
        mean = 3
        std = 1.5
        rddl = f'''
                domain my_test {{
                    pvariables {{ 
                        p : {{ state-fluent,  real, default = 0 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ 
                        p' = Normal({mean}, {std});
                    }}; 
                    reward = 0;
                }}
                non-fluents my_test_empty {{ domain = my_test; }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p')
        import numpy as np
        bins = np.array(conv._normal_bins) * std + mean
        for k, v in zip(bins, conv._normal_probs):
            self.assertEqual(p[k], v)

    def test_normal_var(self):
        mean = 3
        std = 1.5
        rddl = f'''
                domain my_test {{
                    pvariables {{ 
                        p : {{ state-fluent,  real, default = {mean} }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ 
                        p' = Normal(p, p+{std});
                    }}; 
                    reward = 0;
                }}
                non-fluents my_test_empty {{ domain = my_test; }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, mean)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p')
        import numpy as np
        bins = np.array(conv._normal_bins) * (std + p_) + mean
        for k, v in zip(bins, conv._normal_probs):
            self.assertEqual(p[k], v)

    def test_normal_params(self):
        num_bins = 10
        stds = 4
        rddl = f'''
                domain my_test {{
                    pvariables {{ 
                        p : {{ state-fluent,  real, default = 0 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ 
                        p' = Normal(0, 1);
                    }}; 
                    reward = 0;
                }}
                non-fluents my_test_empty {{ domain = my_test; }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter(normal_bins=num_bins, normal_stds=stds)
        conv.convert_str(rddl)
        p_ = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p_, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p')
        import numpy as np
        bins = np.array(conv._normal_bins)
        for k, v in zip(bins, conv._normal_probs):
            self.assertEqual(p[k], v)

    def test_poisson_const(self):
        mean = 10
        rddl = f'''
                domain my_test {{
                    pvariables {{ 
                        p : {{ state-fluent,  real, default = 0 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ 
                        p' = Poisson({mean});
                    }}; 
                    reward = 0;
                }}
                non-fluents my_test_empty {{ domain = my_test; }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p')
        import numpy as np
        bins = np.array(conv._normal_bins) * np.sqrt(conv._poisson_exp_rate) + mean
        for k, v in zip(bins, conv._normal_probs):
            self.assertEqual(p[k], v)

    def test_poisson_var(self):
        mean = 10
        rddl = f'''
                domain my_test {{
                    pvariables {{ 
                        p : {{ state-fluent,  real, default = 0 }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ 
                        p' = Poisson( p * 2 + {mean});
                    }}; 
                    reward = 0;
                }}
                non-fluents my_test_empty {{ domain = my_test; }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, 0)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p')
        import numpy as np
        bins = np.array(conv._normal_bins) * np.sqrt(conv._poisson_exp_rate) + mean
        for k, v in zip(bins, conv._normal_probs):
            self.assertEqual(p[k], v)

    def test_poisson_params(self):
        num_bins = 10
        stds = 4
        expected_rate = 25
        mean = 20
        rddl = f'''
                domain my_test {{
                    requirements {{ 
                        normal_bins{num_bins}, 
                        normal_stds{stds}, 
                        poisson_exp_rate{expected_rate}
                    }};
                    pvariables {{ 
                        p : {{ state-fluent,  real, default = {mean} }};
                        a : {{ action-fluent, bool, default = false }}; 
                    }};
                    cpfs {{ 
                        p' = Poisson(p);
                    }}; 
                    reward = 0;
                }}
                non-fluents my_test_empty {{ domain = my_test; }}
                instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
                '''
        conv = Converter()
        conv.convert_str(rddl)
        p = conv.world.getState(WORLD, 'p', unique=True)
        self.assertEqual(p, mean)
        conv.world.step()
        p = conv.world.getState(WORLD, 'p')
        import numpy as np
        bins = np.array(conv._normal_bins) * np.sqrt(conv._poisson_exp_rate) + mean
        for k, v in zip(bins, conv._normal_probs):
            self.assertEqual(p[k], v)
