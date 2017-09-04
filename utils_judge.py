import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops, math_ops
import timeout
import os, filecmp
import math
import time


'''
ways to run the code:
https://gist.github.com/rajarsheem/6089309fd5bb9b3c71ab
^simplest one
https://github.com/pushkar8723/Aurora/blob/master/Judge/judge.py
https://github.com/kaustubh-karkare/aurora-online-judge/blob/master/judge.py
'''

file_name_incrementor = 0
judge_dir = 'tmp_judge'
length_penalty_weighting = .001
no_compile_penalty = .5
incremental_reward_weighting = .5

def target_id_to_vocab_w_new_line(id_num, target_id_to_vocab):
    if id_num == 8:
        return '\n'
    else:
        return target_id_to_vocab[id_num]

#codes = {200:'success',404:'file not found',400:'error',408:'timeout'}

def create_code_file(program_str, batch_size):
    global file_name_incrementor
    global judge_dir
    if file_name_incrementor > batch_size:
        file_name_incrementor = 0

    file_name = "code_pred_"+str(file_name_incrementor)+".py"
    file_path = os.path.join(judge_dir, file_name)

    if (os.path.isfile(file_path)):
        os.remove(file_path)

    code_pred_file = open(file_path, 'w')
    code_pred_file.write(program_str)

def delete_code_file():
    global file_name_incrementor
    global judge_dir
    #if file_name_incrementor > batch_size:
        #file_name_incrementor = 0

    file_name = "code_pred_"+str(file_name_incrementor)+".py"
    file_path = os.path.join(judge_dir, file_name)

    if (os.path.isfile(file_path)):
        os.remove(file_path)

def run(file, input_file, output_file, test_num, timeout):
    global file_name_incrementor
    global judge_dir
    global incremental_reward_weighting
    timeout = '3' # 3 secs max run time

    file_name = "code_pred_"+str(file_name_incrementor)+".py"
    file_path = os.path.join(judge_dir, file_name)
    path_of_file_generated_code_is_saved_to = file_path

    path_of_test_input_file = input_file
    path_of_returned_output = judge_dir + '/' + 'out' + str(file_name_incrementor) + '_' + str(test_num) + '.txt'

    reward = 0
    #_compile_bool = 
    '''might want to figure out a way to put a memory limit here as well. ulimit? resource?'''
    cmd = 'gtimeout ' + timeout + ' ' + 'python ' + path_of_file_generated_code_is_saved_to + ' < ' + path_of_test_input_file + ' > ' + path_of_returned_output
    r = os.system(cmd)
    if r == 0:
        _compile_bool = True
        c = filecmp.cmp(path_of_returned_output, output_file)
        if c == True:
            '''rewards for subsequent test cases are progressively decreased because codes that pass a few tend to pass the rest'''
            reward+=1/(test_num**(incremental_reward_weighting)) # reward decreases ~logarithmically
            #reward+=1/(test_num) # reward decreases logarithmically
        else:
            pass
    elif r == 256:
        _compile_bool = False
        '''ADD MODULE TO ONE HOT ENCODE ERROR MESSAGE AND REDUCE PENALTY IF MODEL IS ABLE TO PREDICT ERROR MESSAGE'''
        #penalty for not compiling
        #reward = -.5
        '''^I think you can get rid of this line, cause you take of penalty in eval_code'''
    else:
        _compile_bool = True
        _compile_bool = False
        '''^should this one be True or False?'''
        '''does os.system return other functions that should have their own ELIF statements'''
        '''e.g. too much time or too much memory'''
        pass

    if (os.path.isfile(path_of_returned_output)):
        os.remove(path_of_returned_output)

    return reward, _compile_bool

def eval_code(code_pred, test_cases, batch_size, target_id_to_vocab):
    global file_name_incrementor
    global length_penalty_weighting
    global no_compile_penalty
    global incremental_reward_weighting
    length_penalty = length_penalty_weighting * len(code_pred)
    #program_string = "".join([(str(self.target_id_to_vocab_w_new_line(int(target)))) for target in code_pred if int(target) is not 0])
    #[1:-1] removes _GO and _EOS
    
    #program_string = "".join([(str(self.target_id_to_vocab_w_new_line(int(target)))) for target in code_pred[1:-1] if int(target) is not 0])
    #a = [x for x in a if x != 2]
    '''(if int(target) != 0 and int(target) != 1 and int(target) != 2) removes PAD, _GO, & _EOS'''
    program_string = "".join([(str(target_id_to_vocab_w_new_line(int(target), target_id_to_vocab))) for target in code_pred if int(target) != 0 and int(target) != 1 and int(target) != 2])   

    create_code_file(program_string, batch_size)

    max_time = '3' #3 secs

    rewards=0
    max_reward=1e-12  # to avoid division by zero
    for jdx, j in enumerate(test_cases):
        _test_num = jdx+1 # rewards start at 1
        rew, compile_bool = run(None, j[0], j[1], _test_num, 3)
        rewards += rew
        if compile_bool == False:
            break
        #max_reward+=1/(_test_num**(incremental_reward_weighting)) # max reward decreases ~logarithmically

    max_reward += sum(1.0/d**(incremental_reward_weighting) for d in xrange(1,len(test_cases)+1))

    ''' I think reward needs to be whatever sign that is opposite loss; need to check this;'''
    ''' was xent and reinforce opposite signs in Gauthier's code? '''

    '''ah, keep reward positive for now, cause sign is swapped in freinforce loss function'''
    if compile_bool == True:
        total_reward = rewards/max_reward - length_penalty
    else:
        #total_reward = -(no_compile_penalty + length_penalty_weighting)
        total_reward = (-no_compile_penalty + rewards/max_reward - length_penalty) * no_compile_penalty

    #maybe put code here to add successful codes to list of gold programs

    delete_code_file()

    '''if you use parallel futures, you can use thread_id_num instead (of inc) in order to save to diff files'''
    file_name_incrementor += 1

    return total_reward   


def judge_reward(sampled_outputs, test_cases, batch_size, target_id_to_vocab):
    global judge_dir
    if not os.path.exists(judge_dir):
        os.makedirs(judge_dir)

    batch_of_rewards = []
    for i in sampled_outputs:
        batch_of_rewards.append(eval_code(i, test_cases, batch_size, target_id_to_vocab))

        #''' TODO: delete file here '''

        #file_name_incrementor += 1

    return batch_of_rewards

'''IS THERE A WAY TO USE tf.py_func() HERE:
https://www.tensorflow.org/versions/r0.12/api_docs/python/script_ops.html#py_func
'''

'''
_target_id_to_vocab = ['_PAD', '_GO', '_EOS', '_UNK', ' ', '(', ')', '=', '\\n', ':', '[', ']', 'print', ';', ',', '1', 'b', 'c', "'", 'd', '0', 'a', 'e', '.', '+', 'f', '-', 'in', 'g', 'for', 'if', 'h', 'int', 'raw_input', 'i', '*', '2', 'split', 'j', 'else', 'range', 'map', 'k', '>', 'l', '/', '<', 'n', 'len', 'return', '%', 'o', 'm', 'xrange', 'input', 'r', 'append', '3', 'and', 'while', 'def', '!', 'import', 't', 's', 'p', 'elif', '4', '5', 'str', '9', '6', 'max', 'sys', '_', '7', 'sum', 'N', 'break', 'min', 'E', 'or', 'u', 'True', 'S', 'O', 'q', '_a', 'Y', '8', 'not', 'join', '{', '}', 'False', 'stdin', 'list', 'readline', 'from', 'strip', 'lambda', 'count', 'sort', 'y', 'sorted', 'as', 'set', 'w', 'ord', 'abs', '#', 'A', 'x', 'math', 'v', 'I', '\\', 'continue', 'B', 'exit', 'R', 'L', 'C', '__name__', 'bin', 'T', 'index', 'pop', 'float', '&', 'D', 'None', 'z', 'add', 'reverse', 'usr', 'enumerate', 'G', 'write', 're', '^', 'replace', 'P', '?', 'H', 'stdout', 'collections', 'pow', 'get', 'M', 'W', 'key', 'F', 'python', 'sqrt', 'open', 'dict', 'zip', 'find', 'env', 'exec', 'chr', 'coding', 'keys', 'itertools', 'gcd', 'except', 'try', 'U', '|', 'utf', 'is', 'K', 'long', 'global', 'next', '"', 'lower', 'X', 'remove', 'values', 'rstrip', 'del', 'fractions', 'factorial', 'ceil', 'J', 'items', 'reversed', '', 'string', 'Counter', 'format', 'all', 'filter', 'V', 'tuple', 'defaultdict', 'python', 'bisect', 'reduce', 'log', 'assert', 'has_key', 'upper', '`', 'deque', '__author__', 'any', 'insert', 'readlines', '@', 'extend', 'cmp', '$', 'Z', 'operator', 'class', '_b', 'Q', 'read', 'permutations', 'iteritems', 'copy', 'heapq', '__init__', 'file', 'time', 'id', 'floor', 'yield', 'isdigit', 'group', 'heappush', 'sub', 'match', 'combinations', '__future__', 'os', '~', 'close', 'itemgetter', 'findall', 'Queue', 'heappop', 'pi', 'hash', 'popleft', 'startswith', 'isupper', 'quit', 'random', 'round', 'compile', 'Decimal', 'search', 'setdefault', 'eval', '__', 'bisect_left', 'end', 'zfill', 'update', 'setrecursionlimit', 'isalpha', 'ascii_lowercase', 'datetime', 'islower', 'UTF', 'division', '_c', 'print_function', 'ValueError', 'endswith', 'lstrip', 'intersection', 'put', 'sin', 'dir', 'raise', 'argv', 'maxint', 'ln', '__readline', 'KeyError', 'decimal', 'EOFError', 'type', 'object', 'fileinput', 'bisect_right', 'lowercase', 'iter', 'rfind', 'OrderedDict', 'deepcopy', 'divmod', 'product', 'heapify', 'start', 'translate', 'most_common', 'repeat', 'Exception', 'acos', 'swapcase', 'mul', 'value', '__len__', 'size', 'Set', 'Fraction', 'pos', 'izip', 'name', 'buffer', 'bool', 'encoding', 'with', 'path', 'union', 'itervalues', '_data', 'cost', 'groupby', 'rjust', 'data', 'right', 'fabs', 'left', 'fromkeys', 'nextInt', 'hypot', 'idx', 'pprint', '_lcur', 'arr', '_cur', 'io', 'array', 'sets', '__contains__', '_d', 'to', 'num', 'bin', 'test_mode', 'empty', 'imap', 'discard', 'children', 'stderr', 'date', 'splitlines', 'fmod', 'clear', '_ldata', 'numpy', 'level', '__a', 'issubset', 'nextToken', 'islice', 'solve', 'getcontext', 'numbers', 'functools', 'fr', 'randint', 'chain', 'appendleft', 'number', 'calendar', 'partition', 'score', 'cases', 'IndexError', 'PriorityQueue', 'val', 'push', 'StringIO', '__repr__', 'prec', 'ascii_uppercase', 'iterkeys', 'max_count', 'height', 'edges', 'clock', 'groups', '__str__', 'cnt', 'cos', 'ti', 'isspace', 'sons', 'sum_list', 'traceback', 'rank', '__add__', 'pdb', 'length', 'bucket_', 'namedtuple', 'difference', 'weight', 'capacity', 'day', 'label', 'letters', 'finditer', 'answer', 'timedelta', 'mod', 'mask', 'shlex', 'cal', 'hour', 'root', 'begin', 'cycle', 'isEmpty', '_e', 'atan', 'ranks', 'digits', 'logging', '__getitem__', 'asin', 'hex', 'SystemExit', 'status_dict', 'xor', 'parent', 'query', 'strptime', 'graph_dict', 'unichr', 'amount', 'popitem', 'ATK', 'read_int_list', 'cp', 'parents', 'win', 'uppercase', 'stat', 'flush', 'name_array', 'read_int', 'Python', 'rsplit', 'bit_length', 'points', 'StopIteration', 'lose', 'DEF', 'attrgetter', 'month', 'Versions', 'Frameworks', 'char_counts', 'Library', 'nC', 'ans', 'framework', 'front', 'mktime', 'vars', 'le', 'elements', 'container', 'days', 'width', 'seed', 'best', 'readAt', 'maketrans', '__mul__', 'goals', 'testForSquare', 'income', 'go', 'init', 'second', '_convert', 'taxi_no', 'tan', 'milk', 'year', 'base', 'sep', 'frozenset', 'options', 'monthrange', 'rotate', 'peek', 'takewhile', 'eof', 'pts', 'change', 'help', 'rpartition', 'firstpos', 'capitalize', 'st', '__date__', 'isinstance', 'res', 'disks', 'queue', 'heapreplace', 'input_data', 'rindex', 'deg', 'goal', 'lines', 'freeParks', 'INFO', 'rad', 'local', 'basicConfig', 'get_token', 'no', 'point', 'center', 'Find', 'prog', 'scored', 'tHeight', 'imag', 'player', 'max_score', 'dequeue', 'text', '__lt__', 'cc_house', 'node', 'HP', 'pw', 'places', 'nlargest', 'cargo', 'shape', 'dest', 'dfs', 'apples', '__main__', 'pnt', 'wraps', 'repr', 'bytes', 'real', 'current_query', 'used', 'dimensions', 'subtract', 'bytearray', 'prefix', 'getvalue', 'calling_order', 'verbose', 'z_change', 'softspace', 'top', 'zeros', 'counter', 'canCover', 'exp', 'aSteps', 'color', 'unittest', 'complex', 'is_empty', 'select', 'dtype', 'strict', 'head', 'recursive', 'missed', 'nxt', 'graph', 'prev', 'ideal_order', 'minute', 'leader', 'run', 'sus', 'istitle', 'total', 'nums', 'enqueue', 'first', 'oranges', 'merge', 'semifinal', 'energy', 'operation', 'sHeight', 'kind', 'increment', 'methodcaller', 'price', 'rear', 'viewitems', 'urllib', 'sortedprice', 'person', 'position', 'psyco', 'shopsNum', 'get_char_index', 'lch', 'side', 'transaction_dictionary', 'dmg', 'combinations_with_replacement', '__cmp__', 'dif', 'lst', 'nextLine', 'today', 'mnts', 'filepath', 'hi', '_g', '_parent', 'getitem', 'processed', 'bits', 'sell', 'maxh', 'stones', 'fileno', 'secs', 'toprint', 'sub_nodes', 'inp', '__sub__', 'di', 'conjugate', 'weightage', 'issuperset', 'subversion', '_push', '_size', 'area', 'minutes', 'num_house', 'ifilter', 'gap', 'dot', 'calculate', 'is_directed', 'gc', 'tee', 'shops', 'shuffle', 'risk', 'getData', 'restart', '_f', 'update_point', 'cStringIO', 'line', 'out', 'rv', 'miss', '__setitem__', 'visited', 'ascii_letters', 'maxW', 'getResult', 'memo', 'timeit', 'difference_update', 'max_w', '__file__', 'read_prefix', 'exists', 'tree', 'marksSet', 'GBK', 'get_result', '__eq__', 'unicode_literals', 'participant', 'insort', 'dat', 'urlparse', 'mmap', 'nr', 'nonzero', 'step', 'add_edge', 'idt', 'is_integer', 'printAnswer', 'typ', 'pwr', 'format_exc', 'isdisjoint', 'intersection_update', 'gt', 'partial', 'izip_longest', 'sup', 'popen', 'range_', 'compare', 'count_odd', 'acts', 'add_child', 'environ', 'lt', 'from_iterable', 'isalnum', 'st_gid', 'time_started', 'Current', 'actors', 'name_array_len', 'viewvalues', 'nextNode', 'cache', 'dp', 'currentIndex', 'st_uid', 'cumsum', 'happiness', 'wy', 'wx', 'strftime', 'words', 'to_tuple', 'seq_number', 'staticmethod', 'version_info', 'ingridients', 'sentence', 'contents', 'currentLine', 'total_seconds', 'find_path', 'compress', 'subn', 'Add', 'rch', 'neighbors', 'duration', 'drivers', 'cmath', 'getGoal', 'st_ctime', 'gr', 'kameda', 'Take', 'captain', 'woman', 'choice', 'worst', 'ge', '__missing__', 'direc', '__stdin__', 'seconds', 'dep', 'full', 'finally', 'generators', 'nsmallest', 'lostGoal', 'dragon', 'add_vertex', 'oas', 'make_set', 'totalNodes', 'printTree', 'nextRange', 'rec', 'push_to_last', 'presets', 'indices', 'num_node', 'done', 'man', 'outrange', 'cards', 'stay_on', 'rat', 'eq', 'try_bucket', 'denominator', 'demand', 'minDrives', 'order', 'runTheTest', 'number_of_persons', 'read_str', 'cProfile', 'integrate', 'child', 'gbk', 'modulo', '_pop', 'fail', 'yes', 'value_at', 'ctypes', 'Union', 'json', 'Points', 'func', 'add_string', 'fun', 'parse', 'binary', 'isValid', 'max_queue_len', 'build', '_________', 'IOError', 'Arr', 'count_chars', 'GetCost', 'intern', 'home', 'modf', 'utf_', 'numerator', 'ids', 'jun', 'pyenv', 'addRank', 'inc', 'draw', 'float_info', 'pop_out', 'sleep', 'canmoveto', 'IGNORECASE', 'c_int', 'swap_columns', 'childStrings', 'stool', 'access', 'isOut', 'update_energy', 'ACCESS_READ', 'shoe_cost', 'apr', '___', 'nested_scopes', 'abspath', '__call__', 'numher', 'process', 'reset', 'okay', 'disable', '_get_height', 'isleap', 'sh', 'getRight', 'is_full', 'print_tree', 'subprocess', 'NameError', 'total_time', 'timestamp', 'gender', '__and__', 'viewkeys', 'Type', 'rounding', 'get_front', 'nzs', 'lowbit', 'posixpath', 'attack', '____', 'ROUND_DOWN', 'classmethod', 'posix', 'pant_cost', '__or__', 'flags', 'future_builtins', 'colors', 'highest', 'bfs', 'dk', 'vasya', 'getLeft', 'state', 'arrival', 'netGoal', 'omd', 'defense', 'nshops', 'winningindex', 'space_left', 'py', 'indexOf', '__unicode__', 'userid', 'unknownCnt', 'char', 'transactions', 'whitespace_split', 'persons', 'IN', 'getChampion', '__import__', 'absolute_import', 'ndisks', 'groupid', 'py', 'unite', 'bit', 'lost', 'nextString', 'getIntsInLine', 'favCnt', 'name_array_', 'deleted', 'groupdict', 'add_jun', 'winningbid', 'with_statement', 'atoi', 'inside', 'cover', 'ar', 'ends', 'splitline', '__xor__', 'valid', 'skirt_cost', 'age', 'symmetric_difference', 'GetZeroCount', 'MutableString', '_siftdown_max', 'tt', 'cmpC', 'cut_name', 'finish_current_query', 'div', 'ST_GID', 'bfsThrough', 'zs', 'copysign', 'free', 'cmpkey', 'unit', '_siftup_max', '\t', 'calc', 'addTo', 'getmin', 'text_type', '__iter__', 'sample', 'GetSettedBitsCount', 'work', 'setInfo', 'node_type', 'ST_UID', 'push_back_process', 'SyntaxError', 'quantize', 'lucky', 'numOfRituals', 'ST_MTIME', 'title', 'addBox', 'testmod', 'same', 'fac', 'getCopy', 'lo', 'setBegin', 'cut_letter_end', 'doctest', 'isFull', 'now', 'getAnswer', 'build_heap', 'nsets', 'mode', 'bump', 'getLucky', 'addEdge', 'createBST', 'isSupported', 'ret', 'rounds', 'cut_letter_begin', 'or_', 'read_input', 'is_square', 'getInput', 'ljust', 'consume_process', 'return_energy', 'Sum', 'getInfo', 'fromstring', 'disp', 'println', '_element', '__div__', 'apply', 'timetuple', '__hash__', 'connected', 'letter', '_sort', 'getwinner', 'empty_column_index', 'dist', 'make_leader', 'stepping', '__rmul__', '_getwords', 'probB', 'Util', '_heapify_max', 'oct', 'whoisthewinner', 'dx', 'getCityCount', '_round', '_get_max_height', 'isfirstline', 'leter', 'getMaxCount', 'locals', 'GetMaxBit', 'escape', 'default_timer', 'Query', 'read_long', 'insertWord', 'fromiter', 'set_trace', 'minCost', '__rsub__', 'get_unit', '_getchar', 'print_result', 'isPrime', '_getline', 'dup', '_h', 'case', 'insort_left', 'adj', 'create_masks', 'calculate_time', 'appen', 'make', 'solveit', 'remove_front', 'units_left', 'rip_', 'ior', 'remove_rear', 'Update', 'addChild', 'user_text', 'stoolval', 'getmax', 'heappushpop', 'memoizer', 'make_group', 'Position', 'message', 'UserString', 'print_freq', 'output', 'can_be_processed', 'thread', 'beautysubsegment', 'endofword', 'swap_rows', 'add_rear', 'isLucky', 'Crypto', 'numOfNodes', 'write_output', '_getFront', 'dec_char', 'fight', 'decode', 'findMaxHeight', 'getChild', 'linalg', 'frontValue', 'dy', 'dump', 'weekday', 'NUMBER', 'inArea', 'check', 'findways', 'adjacentTo', 'AttributeError', 'normpath', 'atof', 'searchsorted', 'linesep', 'send', 'MutableSet', '_ast', 'pre', 'prize', 'evn', '__ne__', 'notused', 'writelines', 'cmp_to_key', 'InvalidOperation', 'eig', 'hours', 'trunc', 'hashlib', 'threading', 'fsum', 'sqlite', 'Ellipsis', 'localcontext', 'super', 'getrecursionlimit', 'isatty', 'sh', 'tm_yday', 'DivisionByZero', 'astype', 'NotImplemented', 'Calendar', 'argparse', 'seq', 'minimize', 'bash', 'degrees', 'scipy', 'dirname', '_heappushpop_max', 'cursor', 'basestring', 'generate_tokens', 'RLIMIT_STACK', 'pickle', 'setattr', 'STRING', 'ZeroDivisionError', 'phase', 'PrettyPrinter', '__slots__', 'frexp', '_make', '_bisect', 'timegm', 'month_name', 'target', 'resource', 'sha', 'tokenize', 'argmax', 'bounds', 'getopt', 'difflib', 'tt', 'bincount', 'stack_size', 'print_exc', 'debugg', 'globals', 'args', 'ImportError', '__mod__', 'lgamma', 'KeyboardInterrupt', 'indent', 'fill', 'matrix_power', 'user', 'setrlimit', 'utf', '__gt__', 'POP', 'unicode', 'colorama', 'Thread', '__reversed__', 'arange', 'netloc', 'radians', 'logical_and', 'fillvalue', 'assertSequenceEqual', 'printable', 'pyton', '_container', 'optimize', 'maxlen', 'update_wrapper', 'isoformat', 'Num', 'span', 'suf', 'maxsize', 'TestCase', 'total_ordering', 'dropwhile', 'SequenceMatcher', 'up', 'connect', 'amax', 'task_done', 'align', 'hexdigest', 'monthdayscalendar']
batch_size=6
sampled_outputs=[[1, 21, 7, 33, 5, 6, 13, 12, 4, 108, 5, 21, 6, 2, 0, 0, 0, 0, 0],[1, 21, 7, 33, 5, 6, 13, 12, 4, 108, 5, 21, 6, 2, 0, 0, 0, 0, 0]]
_test_cases = [['tmp_judge/3_B/samples/1_input.txt', 'tmp_judge/3_B/samples/1_output.txt'], ['tmp_judge/3_B/samples/2_input.txt', 'tmp_judge/3_B/samples/2_output.txt'], ['tmp_judge/3_B/samples/3_input.txt', 'tmp_judge/3_B/samples/3_output.txt']]
start = time.time()
judge_reward(sampled_outputs, _test_cases, batch_size, _target_id_to_vocab)
end = time.time()
print(end - start)
#'''

