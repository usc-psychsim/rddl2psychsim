from pyrddl.parser import RDDLParser
from pyrddl.rddl import RDDL

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def parse_rddl_file(rddl_file: str, verbose: bool) -> RDDL:
    # read RDDL file
    with open(rddl_file, 'r') as file:
        rddl = file.read()

    return parse_rddl(rddl, verbose)


def parse_rddl(rddl: str, verbose: bool) -> RDDL:
    # parse RDDL
    rddl_parser = RDDLParser(verbose=verbose)
    # rddl_parser.debugging = verbose
    rddl_parser.build()
    return rddl_parser.parse(rddl)
