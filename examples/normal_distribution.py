import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from psychsim.pwl import WORLD
from rddl2psychsim.conversion.converter import Converter

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

COVERED_STDS = 3
NUM_BINS = 7
MU = 3
STD = 2
NUM_SAMPLES = 10000

RDDL = f'''
domain my_test {{
    pvariables {{ 
        p : {{ state-fluent,  real, default = 0 }};
        a : {{ action-fluent, bool, default = false }}; 
    }};
    cpfs {{ 
        p' = Normal({MU}, {STD});
    }}; 
    reward = 0;
    }}
    non-fluents my_test_empty {{ domain = my_test; }}
    instance my_test_inst {{ domain = my_test; init-state {{ a; }}; }}
'''

if __name__ == '__main__':
    conv = Converter(normal_bins=NUM_BINS, normal_stds=COVERED_STDS)
    conv.convert_str(RDDL)

    samples = []
    for i in range(NUM_SAMPLES):
        conv.world.step(select=True)  # select sample
        p = conv.world.getState(WORLD, 'p', unique=True)
        samples.append(p)
    mu, std = stats.norm.fit(samples)
    print(f'estimated mu={mu}, std={std}')

    plt.figure()
    bins = np.linspace(-COVERED_STDS, COVERED_STDS, 100) * STD + MU
    plt.plot(bins, stats.norm.pdf(bins, loc=MU, scale=STD),
             'r-', lw=5, alpha=0.6, label='original pdf')
    plt.plot(bins, stats.norm.pdf(bins, loc=mu, scale=std),
             'k--', lw=2, label='estimated pdf')
    plt.legend()
    plt.show()
