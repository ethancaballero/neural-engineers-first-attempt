#!/usr/bin/env python
# -*- coding: utf-8 -*-

# altered from github.com/IndicoDataSolutions/Passage/blob/master/passage/preprocessing.py#L37

import string

puncs_and_digits = set(string.punctuation+string.digits)
puncs_and_digits.remove('_')
puncs_and_digits.add('\n')
puncs_and_digits.add('\t')
puncs_and_digits.add(u'’')
puncs_and_digits.add(u'‘')
puncs_and_digits.add(u'“')
puncs_and_digits.add(u'”')
puncs_and_digits.add(u'´')
puncs_and_digits.add(u"'")
puncs_and_digits.add(u'"')
puncs_and_digits.add('')

"""
p1 = '#', '!', '/', 'usr', '/', 'bin', '/', 'env', ' ', 'python', '2', '.', '6', '\n'
p2 = '#', '!', '/', 'usr', '/', 'bin', '/', 'env', ' ', 'python', '2', '.', ‘7, '\n'
p3 = '#', '!', '/', 'usr', '/', 'bin', '/', 'env', ' ', 'python', '2', '\n'
p4 = '#', '!', '/', 'usr', '/', 'bin', '/', 'env', ' ', 'python', '\n'
"""

#!/usr/bin/env python2.6\n

p1 = '#!/usr/bin/env python2.7\n'
p2 = '#!/usr/bin/env python2.6\n'
p3 = '#!/usr/bin/env python2\n'
p4 = '#!/usr/bin/env python\n'

def tokenize_python_code(text):
    string_mode = False
    single_quote_mode = False
    double_quote_mode = False    
    tokenized = []
    text.replace('\r\n', '\n').replace(p1, '').replace(p2, '').replace(p3, '').replace(p4, '')
    w = ''
    for t in text:
        #print w
        #single_quote_toggle
        if t == "'":
            if double_quote_mode == False:
                    string_mode = not string_mode
                    single_quote_mode = not single_quote_mode
            else:
                pass
        #double_quote_toggle
        if t == '"':
            if single_quote_mode == False:
                string_mode = not string_mode
                double_quote_mode = not double_quote_mode
            else:
                pass
        if string_mode == True:
            tokenized.append(w)
            tokenized.append(t)
            w = ''
        elif t in puncs_and_digits:
            tokenized.append(w)
            tokenized.append(t)
            w = ''
        elif t == ' ':
            tokenized.append(w)
            tokenized.append(t)
            w = ''
        else:
            w += t
    if w != '':
        tokenized.append(w)
    tokenized = [token for token in tokenized if token]
    return tokenized