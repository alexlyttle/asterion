"""
parser.py
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args(description, defaults={}):
    d = {'e': 0.1, 'f': '.3f', 'l': 0.1, 'n': 100}
    for key, value in defaults.items():
        d[key] = value
    
    parser = ArgumentParser(
        description=description, 
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-e', '--error', metavar='ERR', type=float,
                        default=d['e'], help='error (noise) applied to targets')
    parser.add_argument('-f', '--format', metavar='FMT', type=str,
                        default=d['f'], help='output float format ')
    parser.add_argument('-l', '--lrate', metavar='LRT', type=float,
                        default=d['l'], help='learning rate')
    parser.add_argument('-n', '--numsteps', metavar='N', type=int,
                        default=d['n'], help='number of steps to train')
    parser.add_argument('-p', '--showplots', action='store_true',
                        help='show plots (requires matplotlib)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable verbose output')
    args = parser.parse_args()
    return args
