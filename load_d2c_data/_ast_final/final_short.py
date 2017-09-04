# -*- coding: utf-8 -*-

'''taken from this URL: https://docs.python.org/2/library/ast.html'''
'''AbstractProgramSet class design from https://github.com/mokemokechicken/keras_npi'''

'''Star takes/transforms multiple parameters as/into a list, passing no parameters to star results in "[]" '''
'''Question Mark takes/transforms multiple parameters as/into a tuple, passing no parameters to star results in "None" '''
'''No ?/* takes exactly one argument'''

import string
import ast
from ast import *
from astmonkey import visitors, transformers
import re
import numpy as np
import collections
from collections import OrderedDict

class Program:
    output_to_env = False

    def __init__(self, name, *args):
        self.name = name
        self.args = args
        #print self.args
        self.program_id = None

        '''self.variadic_args = [list containing names of which arguments are variadic (can be None if None are variadic)]'''

    def description_with_args(self, args):
        int_args = args.decode_all()
        return "%s(%s)" % (self.name, ", ".join([str(x) for x in int_args]))

    def to_one_hot(self, size, dtype=np.float):
        ret = np.zeros((size,), dtype=dtype)
        ret[self.program_id] = 1
        return ret

    def do(self, env, args):
        raise NotImplementedError()

    def __str__(self):
        '''
        return "<Program: name=%s>" % self.name
        '''
        return self.name

class AbstractProgramSet:

    def __init__(self):
        self.program_map = {}
        self.program_id = 0

    def register(self, pg):
        pg.program_id = self.program_id
        self.program_map[pg.program_id] = pg
        self.program_id += 1

    def create_and_register_all(self, all_pgs_in_class):
        for i in all_pgs_in_class:
            self.register(Program(i, *all_pgs_in_class[i]))

    def get(self, i):
        return self.program_map.get(i)

    def get_arguments(self, i):
        return self.program_map.get(i).args


mod_dict = {"Module": ['stmt* body'],
            "Interactive": ['stmt* body'],
            "Expression": ['expr body'],
            "Suite": ['stmt* body'],
            }

stmt_dict = {"FunctionDef": ['identifier name', 'arguments args', 'stmt* body', 'expr* decorator_list'],
            "ClassDef": ['identifier name', 'expr* bases', 'stmt* body', 'expr* decorator_list'],
            "Return": ['expr? value'],
            "Delete": ['expr* targets'],
            "Assign": ['expr* targets', 'expr value'],
            "AugAssign": ['expr target', 'operator op', 'expr value'],
            "Print": ['expr? dest', 'expr* values', 'bool nl'],
            "For": ['expr target', 'expr iter', 'stmt* body', 'stmt* orelse'],
            "While": ['expr test', 'stmt* body', 'stmt* orelse'],
            "If": ['expr test', 'stmt* body', 'stmt* orelse'],
            "With": ['expr context_expr', 'expr? optional_vars', 'stmt* body'],
            "Raise": ['expr? type', 'expr? inst', 'expr? tback'],
            "TryExcept": ['stmt* body', 'excepthandler* handlers', 'stmt* orelse'],
            "TryFinally": ['stmt* body', 'stmt* finalbody'],
            "Assert": ['expr test', 'expr? msg'],
            "Import": ['alias* names'],
            "ImportFrom": ['identifier? module', 'alias* names', 'int? level'],
            "Exec": ['expr body', 'expr? globals', 'expr? locals'],
            "Global": ['identifier* names'],
            "Expr": ['expr value'],
            "Pass": [],
            "Break": [],
            #"attributes": ['int lineno', 'int col_offset'],
            }

expr_dict = {"BoolOp": ['boolop op', 'expr* values'],
            "BinOp": ['expr left', 'operator op', 'expr right'],
            "UnaryOp": ['unaryop op', 'expr operand'],
            "Lambda": ['arguments args', 'expr body'],
            "IfExp": ['expr test', 'expr body', 'expr orelse'],
            "Dict": ['expr* keys', 'expr* values'],
            "Set": ['expr* elts'],
            "ListComp": ['expr elt', 'comprehension* generators'],
            "SetComp": ['expr elt', 'comprehension* generators'],
            "DictComp": ['expr key', 'expr value', 'comprehension* generators'],
            "GeneratorExp": ['expr elt', 'comprehension* generators'],
            "Yield": ['expr? value'],
            "Compare": ['expr left', 'cmpop* ops', 'expr* comparators'],
            "Call": ['expr func', 'expr* args', 'keyword* keywords, expr? starargs', 'expr? kwargs'],
            "Repr": ['expr value'],
            "Num": ['object n'],
            "Str": ['string s'],
            "Attribute": ['expr value', 'identifier attr', 'expr_context ctx'],
            "Subscript": ['expr value', 'slice slice', 'expr_context ctx'],
            "Name": ['identifier id', 'expr_context ctx'],
            "List": ['expr* elts', 'expr_context ctx'], 
            "Tuple": ['expr* elts', 'expr_context ctx'],
            #"attributes": ['int lineno', 'int col_offset'],
            }

expr_context_dict = {"Load": [], "Store": [], "Del": [], "AugLoad": [], "AugStore": [], "Param": []}

slice_dict = {"Ellipsis": [], 
        "Slice": ['expr? lower', 'expr? upper', 'expr? step'],
        "ExtSlice": ['slice* dims'],
        "Index": ['expr value'],
        }

boolop_dict = {"And": [], "Or": []}

operator_dict = {"Add": [], "Sub": [], "Mult": [], "Div": [], "Mod": [], 
            "Pow": [], "LShift": [], "RShift": [], "BitOr": [], 
            "BitXor": [], "BitAnd": [], "FloorDiv": []}

unaryop_dict = {"Invert": [], "Not": [], "UAdd": [], "USub": []}

cmpop_dict = {"Eq": [], "NotEq": [], "Lt": [], "LtE": [], "Gt": [], 
            "GtE": [], "Is": [], "IsNot": [], "In": [], "NotIn": []}

comprehension_dict = {"comprehension": ['expr target', 'expr iter', 'expr* ifs']}

excepthandler_dict = {"ExceptHandler": ['expr? type', 'expr? name', 'stmt* body'], 
                "attributes ": ['int lineno', 'int col_offset']}

arguments_dict = {"arguments": ['expr* args', 'identifier? vararg', 'identifier? kwarg', 'expr* defaults']}

keyword_dict = {"keyword": ['identifier arg', 'expr value']}

alias_dict = {"alias": ['identifier name', 'identifier? asname']}

bool_dict = '''True/False'''


'''below are built_in_types that skip the class and jump straight to function'''
'''e.g. looks like ='turnt' or =12 instead of looking like =Load() or =Store()'''
'''if one these, rnn needs to generate or copy a sequence of character'''

#identifier
#int
#string
#object
#bool  #selects True or False


#ones that take string with quotes are string, identifier
#string can be any character(s)
#identifier can only start with letter and _ and can only consist of those and num
#ones that take string without quotes are int, object
#int can only be numbers, 'L', '-'
#object can only be numbers, 'L' decimal_point, 'e', '+', '-'


program_classes_list = ['mod', 'stmt', 'expr', 'expr_context', 'slice', 'boolop', 'operator', 'unaryop', 'cmpop', 'comprehension', 'excepthandler', 'arguments', 'keyword', 'alias']

program_classes = {}

'''THIS LOOP SHOULD PROBABLY BE PUT IN A CLASS OR FUNCTION'''
for i in program_classes_list:

    program_classes[i] = AbstractProgramSet()

    exec("program_classes[i].create_and_register_all(%s)" % (i+"_dict"))

def to_one_hot(program_id, size, dtype=np.float):
    ret = np.zeros((size,), dtype=dtype)
    ret[program_id] = 1
    return ret

def get_pc_index(key_name):
    for i in range(len(program_classes.keys())):
        if program_classes.keys()[i] == str(key_name):
            return i

def get_p_index(pc_name, p_name):
    for i in range(0, program_classes[pc_name].program_id):
        if str(program_classes[pc_name].get(i)) == str(p_name):
            return i

args_demo = ('expr* args', 'identifier? vararg', 'identifier? kwarg', 'expr* defaults')

arg_amount_dic = {1: '', 2: '*', 3: '?'}


'''I think this is used to let rnn know how many arguments it's selecting'''
def argument_parser(arguments):
    parse_args =[]
    for argument in arguments:
        parse_args.append(re.split(' ', argument))
    return parse_args

#print argument_parser(args_demo)
