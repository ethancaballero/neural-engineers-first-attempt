# -*- coding: utf-8 -*-

'''taken from this URL: https://docs.python.org/2/library/ast.html'''

'''Star takes/transforms multiple parameters as/into a list, passing no parameters to star results in "[]" '''
#'''Question Mark takes/transforms multiple parameters as/into a tuple, passing no parameters to star results in "None" '''
'''Question Mark takes 1 or less parameters, passing no parameters to star results in "None" '''
'''If Question Mark receives multiple parameters/arguments, it calls a tuple function to place multiple in'''
'''No ?/* takes exactly one argument'''


from final_short import *
import fileinput
import re
import meta
import minipy


def get_char_dict():
    chars = string.printable

    char_to_id = {}
    char_id = 0

    for i in chars:
        char_to_id[i] = char_id
        char_id += 1
    
    return char_to_id

#print get_char_dict()
#asdf

def get_char_dict_for_all_in_statement_Str(inp):
    out = 'Str('

    #id of Str
    out = '39,'
    char_dict = get_char_dict()
    for i in inp:
        out += str(char_dict[i] + 99) + ','

    return out

def get_char_dict_for_all_in_statement_no_quote(inp):
    out = ','

    char_dict = get_char_dict()
    for i in inp:
        out += str(char_dict[i] + 99) + ','

    return out

def get_char_dict_for_all_in_statement_quote(inp):
    #id of single quote
    out = ",201,"

    char_dict = get_char_dict()
    for i in inp:
        out += str(char_dict[i] + 99) + ','

    out += ",98"
    return out



chars_to_ids = get_char_dict()
ids_to_chars = dict((v, k) for k, v in chars_to_ids.iteritems())

which_dict = {'Raise': 'stmt_dict', 'Param': 'expr_context_dict', 'IsNot': 'cmpop_dict', 'Suite': 'mod_dict', 'Exec': 'stmt_dict', 'FloorDiv': 'operator_dict', 'TryFinally': 'stmt_dict', 'Not': 'unaryop_dict', 'Lambda': 'expr_dict', 'Mult': 'operator_dict', 'Mod': 'operator_dict', 'Load': 'expr_context_dict', 'BoolOp': 'expr_dict', 'Yield': 'expr_dict', 'While': 'stmt_dict', 'Div': 'operator_dict', 'Or': 'boolop_dict', 'FunctionDef': 'stmt_dict', 'Gt': 'cmpop_dict', 'Global': 'stmt_dict', 'Index': 'slice_dict', 'Sub': 'operator_dict', 'For': 'stmt_dict', 'UnaryOp': 'expr_dict', 'Invert': 'unaryop_dict', 'NotIn': 'cmpop_dict', 'DictComp': 'expr_dict', 'LShift': 'operator_dict', 'Ellipsis': 'slice_dict', 'Print': 'stmt_dict', 'Subscript': 'expr_dict', 'BitOr': 'operator_dict', 'ExceptHandler': 'excepthandler_dict', 'ClassDef': 'stmt_dict', 'Delete': 'stmt_dict', 'NotEq': 'cmpop_dict', 'LtE': 'cmpop_dict', 'Pass': 'stmt_dict', 'attributes': 'expr_dict', 'Eq': 'cmpop_dict', 'Add': 'operator_dict', 'comprehension': 'comprehension_dict', 'Import': 'stmt_dict', 'TryExcept': 'stmt_dict', 'Store': 'expr_context_dict', 'GtE': 'cmpop_dict', 'Tuple': 'expr_dict', 'Break': 'stmt_dict', 'USub': 'unaryop_dict', 'SetComp': 'expr_dict', 'Del': 'expr_context_dict', 'Str': 'expr_dict', 'Expression': 'mod_dict', 'Assign': 'stmt_dict', 'Interactive': 'mod_dict', 'And': 'boolop_dict', 'ExtSlice': 'slice_dict', 'Compare': 'expr_dict', 'Set': 'expr_dict', 'keyword': 'keyword_dict', 'Attribute': 'expr_dict', 'Num': 'expr_dict', 'Call': 'expr_dict', 'Slice': 'slice_dict', 'Lt': 'cmpop_dict', 'Dict': 'expr_dict', 'AugLoad': 'expr_context_dict', 'Return': 'stmt_dict', 'arguments': 'arguments', 'Repr': 'expr_dict', 'Assert': 'stmt_dict', 'attributes ': 'excepthandler_dict', 'ImportFrom': 'stmt_dict', 'UAdd': 'unaryop_dict', 'With': 'stmt_dict', 'AugAssign': 'stmt_dict', 'RShift': 'operator_dict', 'Name': 'expr_dict', 'BinOp': 'expr_dict', 'Expr': 'stmt_dict', 'List': 'expr_dict', 'BitXor': 'operator_dict', 'Pow': 'operator_dict', 'Is': 'cmpop_dict', 'BitAnd': 'operator_dict', 'Module': 'mod_dict', 'AugStore': 'expr_context_dict', 'alias': 'alias_dict', 'In': 'cmpop_dict', 'If': 'stmt_dict', 'GeneratorExp': 'expr_dict', 'ListComp': 'expr_dict', 'IfExp': 'expr_dict'}
which = {'Raise': 'stmt', 'Param': 'expr_context', 'IsNot': 'cmpop', 'Suite': 'mod', 'Exec': 'stmt', 'FloorDiv': 'operator', 'TryFinally': 'stmt', 'Not': 'unaryop', 'Lambda': 'expr', 'Mult': 'operator', 'Mod': 'operator', 'Load': 'expr_context', 'BoolOp': 'expr', 'Yield': 'expr', 'While': 'stmt', 'Div': 'operator', 'Or': 'boolop', 'FunctionDef': 'stmt', 'Gt': 'cmpop', 'Global': 'stmt', 'Index': 'slice', 'Sub': 'operator', 'For': 'stmt', 'UnaryOp': 'expr', 'Invert': 'unaryop', 'NotIn': 'cmpop', 'DictComp': 'expr', 'LShift': 'operator', 'Ellipsis': 'slice', 'Print': 'stmt', 'Subscript': 'expr', 'BitOr': 'operator', 'ExceptHandler': 'excepthandler', 'ClassDef': 'stmt', 'Delete': 'stmt', 'NotEq': 'cmpop', 'LtE': 'cmpop', 'Pass': 'stmt', 'attributes': 'expr', 'Eq': 'cmpop', 'Add': 'operator', 'comprehension': 'comprehension', 'Import': 'stmt', 'TryExcept': 'stmt', 'Store': 'expr_context', 'GtE': 'cmpop', 'Tuple': 'expr', 'Break': 'stmt', 'USub': 'unaryop', 'SetComp': 'expr', 'Del': 'expr_context', 'Str': 'expr', 'Expression': 'mod', 'Assign': 'stmt', 'Interactive': 'mod', 'And': 'boolop', 'ExtSlice': 'slice', 'Compare': 'expr', 'Set': 'expr', 'keyword': 'keyword', 'Attribute': 'expr', 'Num': 'expr', 'Call': 'expr', 'Slice': 'slice', 'Lt': 'cmpop', 'Dict': 'expr', 'AugLoad': 'expr_context', 'Return': 'stmt', 'arguments': 'arguments', 'Repr': 'expr', 'Assert': 'stmt', 'attributes ': 'excepthandler', 'ImportFrom': 'stmt', 'UAdd': 'unaryop', 'With': 'stmt', 'AugAssign': 'stmt', 'RShift': 'operator', 'Name': 'expr', 'BinOp': 'expr', 'Expr': 'stmt', 'List': 'expr', 'BitXor': 'operator', 'Pow': 'operator', 'Is': 'cmpop', 'BitAnd': 'operator', 'Module': 'mod', 'AugStore': 'expr_context', 'alias': 'alias', 'In': 'cmpop', 'If': 'stmt', 'GeneratorExp': 'expr', 'ListComp': 'expr', 'IfExp': 'expr'}

ast_source='''Module([Print(None, [Num(1), Num(2)], True), Print(None, [Num(2)], True)])'''

#print raw_python

def minify_python(filename):
	mini_kwargs = {'rename': True, 'preserve': None, 'indent': 1, 'selftest': True, 'joinlines': True, 'docstrings': True, 'debug': False}
	return minipy.minify(filename, **mini_kwargs)


mini = minify_python('test_text.txt')
print mini


def raw_python_to_ast(raw_python_source):
	node = parse(raw_python_source)
	node = transformers.ParentNodeTransformer().visit(node)
	ast_graph = meta.asttools.str_ast(node, indent='', newline='')

	#"""
	print ast_graph
	'''^could return here, but field can't be removed for some reason'''
	print meta.asttools.python_source(node)
	#"""

	node_from_scratch = eval(ast_graph)
	fixed = fix_missing_locations(node_from_scratch)
	#print dump(fixed, annotate_fields=True, include_attributes=False)

	'''
	print dump(fixed, annotate_fields=False, include_attributes=False)
	print eval(compile(fixed, 'compiled_code', 'exec'))
	#'''

	return dump(fixed, annotate_fields=False, include_attributes=False)

#print raw_python_to_ast(raw_python)

#ast_source = raw_python_to_ast(raw_python)
ast_source = raw_python_to_ast(mini)


def get_idx_in_pc(index_in_all):
	if index_in_all < 4:
		index_in_pc = index_in_all
		func_name = program_classes['mod'].get(index_in_pc)
	elif index_in_all < 27:
		index_in_pc = index_in_all - 4
		func_name = program_classes['stmt'].get(index_in_pc)
	elif index_in_all < 50:
		index_in_pc = index_in_all - 27
		func_name = program_classes['expr'].get(index_in_pc)
	elif index_in_all < 56:
		index_in_pc = index_in_all - 50
		func_name = program_classes['expr_context'].get(index_in_pc)
	elif index_in_all < 60:
		index_in_pc = index_in_all - 56
		func_name = program_classes['slice'].get(index_in_pc)
	elif index_in_all < 62:
		index_in_pc = index_in_all - 60
		func_name = program_classes['boolop'].get(index_in_pc)
	elif index_in_all < 74:
		index_in_pc = index_in_all - 62
		func_name = program_classes['operator'].get(index_in_pc)
	elif index_in_all < 78:
		index_in_pc = index_in_all - 74
		func_name = program_classes['unaryop'].get(index_in_pc)
	elif index_in_all < 88:
		index_in_pc = index_in_all - 78
		func_name = program_classes['cmpop'].get(index_in_pc)
	elif index_in_all < 89:
		index_in_pc = index_in_all - 88
		func_name = program_classes['comprehension'].get(index_in_pc)
	elif index_in_all < 91:
		index_in_pc = index_in_all - 89
		func_name = program_classes['excepthandler'].get(index_in_pc)
	elif index_in_all < 92:
		index_in_pc = index_in_all - 91
		func_name = program_classes['arguments'].get(index_in_pc)
	elif index_in_all < 93:
		index_in_pc = index_in_all - 92
		func_name = program_classes['keyword'].get(index_in_pc)
	elif index_in_all < 94:
		index_in_pc = index_in_all - 93
		func_name = program_classes['alias'].get(index_in_pc)
	else:
		pass

	return func_name

#'''
print ast_source
#'''

def ast_to_linearized_ids(source):
	new_l=[]

	source = re.sub(r'Str\((.*?)\)', lambda x: get_char_dict_for_all_in_statement_Str(x.group(1)), source)

	#source = re.sub(r'\ \(([\s]*?)', '(, 103, ', source)
	source = re.sub(r'\[([\s]*?)', '[, 199, ', source)

	#source = re.sub(r'\)([\s]*?)', '), 94, ,', source)
	source = re.sub(r'\]([\s]*?)', '], 94, ,', source)
	source = re.sub(r'\)([\s]*?)', '), 200, ,', source)

	# L explained: https://docs.python.org/2/tutorial/floatingpoint.html
	source = re.sub(r'\(([0-9\.\+\e\L]*?)\)', lambda x: get_char_dict_for_all_in_statement_no_quote(x.group(1)), source)
	source = re.sub(r'\((\-[0-9\.+\e\L]*?)\)', lambda x: get_char_dict_for_all_in_statement_no_quote(x.group(1)), source)
	source = re.sub(r'\,\ ([0-9\.+\e\L]*?)\)', lambda x: get_char_dict_for_all_in_statement_no_quote(x.group(1)), source)
	source = re.sub(r'\,\ (\-[0-9\.+\e\L]*?)\)', lambda x: get_char_dict_for_all_in_statement_no_quote(x.group(1)), source)

	'''ADD MODULE/FLAG TO CHANGE ALL IDENTIFIER STRINGS TO GENERIC NAMES'''
	source = re.sub(r'\'([A-Za-z0-9_]*?)\'', lambda x: get_char_dict_for_all_in_statement_quote(x.group(1)), source)

	source = re.sub(r'\[([A-Za-z]*?)\(', '[\\1,(', source)
	source = re.sub(r'\(([A-Za-z]*?)\,', '(\\1,', source)
	source = re.sub(r'\,([\sA-Za-z]*?)\(', '\\1(', source)
	source = re.sub(r'\,([\sA-Za-z]*?)\)', '\\1)', source)

	source = source.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')
	source = source.replace('Module  ', '[')
	source += ']'
	source = source.replace('    ', '   ').replace('   ', '  ').replace('  ', ' ')
	source = source.replace(',,', ',')
	source = source.replace(' ,', ',')
	source = source.replace(' ', ', ')
	source = source.replace(',,', ',')
	source = source.replace(' ,', '')
	source = source.replace(',,', ',')
	source = source.replace(' ,', '')

	'''might need to transfer cross entropy array to diffrent file so that ast functions with same name don't interfere halfway'''

	def replace_if_in_dict(i, which_dict):
		print i
		if i in which_dict:
			return which_dict[i]
		else:
			return ''

	source = source.replace(', ',',').replace('[','').replace(',]','').replace(', ]','').replace(']','').split(',')

	for idx, i in enumerate(source):
		if i == "''":
			del source[idx]
		else:
			pass

	add_to_give_each_func_unique_id = {'arguments': 91, 'slice': 56, 'cmpop': 78, 'expr_context': 50, 'keyword': 92, 'unaryop': 74, 'expr': 27, 'boolop': 60, 'stmt': 4, 'excepthandler': 89, 'alias': 93, 'comprehension': 88, 'operator': 62, 'mod': 0}

	source = [get_p_index(which[i], i) + add_to_give_each_func_unique_id[which[i]] if i in which else '95' if i == 'None' else '96' if i == 'True' else '97' if i == 'False' else i for i in source]

	'''SEE HOW LATENT PREDICTOR NETWORKS FOR CODE GENERATION 1603.06744 HANDLED CROSS ENTROPY OF WORDS_IDS & CHARACTERS SIMULTANEUSLY'''

	if source[0] == '\n':
		del source[0]

	if source[-1] == '\n':
		del source[-1]

	for idx, i in enumerate(source):
		if source[idx] == '':
			del source[idx]

	for idx, i in enumerate(source):
		if not isinstance( source[idx], int ): # True
			#print source[idx]
			#print eval(source[idx])
			source[idx] = eval(source[idx])

	return source

linearized_ids_of_ast = ast_to_linearized_ids(ast_source)

print linearized_ids_of_ast

def convert_ids_to_ast(source_as_ids):
	to_ast = ''
	to_ast += 'Module('

	string_mode = False
	single_quote_mode = False
	double_quote_mode = False

	for idx, i in enumerate(source_as_ids):
		if i == 5000:
			#originally 103; definately getting rid of
			to_ast += str('(')
		elif i == 199:
			to_ast += str('[')
		elif i == 94:
			to_ast += str('], ')

		elif i == 200:
			to_ast += str('), ')

		#this is for closing identifier quotes; identifier can only start with letter and _ and can only consist of those and num
		elif i == 201:
			to_ast += str("'")
		elif i == 98:
			to_ast += str("', ")

		elif i < 95:
			to_ast += str(get_idx_in_pc(i)) + str('(')
			#Str == 39 
			if i == 39:
				string_mode = not string_mode

		elif i > 98 and i < 199:
			to_ast += str(ids_to_chars[i-99])

			#single_quote == 167
			if i == 167:
				if double_quote_mode == False:
					single_quote_mode = not single_quote_mode 
					if single_quote_mode == False:
						string_mode = not string_mode
						if string_mode == False:
							to_ast += str('), ')

			#double_quote == 162
			if i == 162:
				if single_quote_mode == False:
					double_quote_mode = not double_quote_mode
					if double_quote_mode == False:
						string_mode = not string_mode
						if string_mode == False:
							to_ast += str('), ')

		elif i == 95:
			to_ast += 'None, '
		elif i == 96:
			to_ast += 'True, '
		elif i == 97:
			to_ast += 'False, '

		else:
			to_ast += str(i)

	return to_ast[0:-2]


#'''
print convert_ids_to_ast(linearized_ids_of_ast)
#'''

'''
YOU NEED TO ADD ONE SO THAT ZERO IS NOT AN ID
OR MAYBE NOT SINCE MODULE WILL BE ZERO ID IF ORDERED DICT IS USED
IF SO, STILL NEED NEED TO REMOVE MODULE FROM IDS I THINK

Module (not used I think) = 0
function_ids = 1-93
] = 94
None = 95
True = 96
False = 97
"', " = 98	#this closes/end identifier char sequence
chars = 99-198 (len of char vocab is literally 100)

[ = 199
^is there a way to dyamically add this like you did with (

elif i == 200:
	to_ast += str('), ')

"'" = 201

202 is not used, get rid of it; it's original purpose was to end string but second quote now does that. 

'''
'''These will be skipped by function sum loop
199 [ is automatically added at beginning of arguments for Star
200 ) is automatically added when all parameters of function are filled
201 ' (<--different id than ' id) is automatically called when identifier is called
so will be skipped by loss function sum loop
'''
'''
"None" is appended as possible funtion id to select from if question_mark
"]" is is appended as possible funtion id to select from if asterisk
'''
'''
identifier rnn can only start with letter and _ and can only consist of those and num
string rnn can select any character, although starts and ends with single or double quote (is there a way to mask out vocab for this first step)

object rnn can only take ints and decimal point (period)
int can only take ints, althogh i'm not sure if its ever used
'''


