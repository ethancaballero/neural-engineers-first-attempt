# -*- coding: utf-8 -*-




import os
import minipy
from tokenize_python import tokenize_python_code
from question_to_fields_final_perfect import question_to_tokenized_fields

# rootdir1 = '/Users/ethancaballero/Neural-Engineer_Candidates/dmn-tf-alter_working_decoder_d2c/load_d2c_data/description2code_current/codeforces_delete'
# rootdir2 = '/Users/ethancaballero/Neural-Engineer_Candidates/dmn-tf-alter_working_decoder_d2c/load_d2c_data/description2code_current/hackerearth/problems_college'
# rootdir3 = '/Users/ethancaballero/Neural-Engineer_Candidates/dmn-tf-alter_working_decoder_d2c/load_d2c_data/description2code_current/hackerearth/problems_normal'


rootdir1 = 'load_d2c_data/description2code_current/codeforces_delete'
rootdir2 = 'load_d2c_data/description2code_current/hackerearth/problems_college'
rootdir3 = 'load_d2c_data/description2code_current/hackerearth/problems_normal'

#rootdir1 = '.../description2code_current/codeforces_delete'
#rootdir2 = '.../description2code_current/hackerearth/problems_college'
#rootdir3 = '.../description2code_current/hackerearth/problems_normal'

#print os.path.abspath(os.path.join('.', os.pardir))
#print os.path.abspath(os.path.join('.', os.curdir)) + '/weights'
#asdf

#rootdir = '/Users/ethancaballero/Neural-Engineer_Candidates/tokenize_experiment/yads_data_loader/description2code_current/blah'

main_count = 0
sub_count = 0
#questions_count = 3000
questions_count = 3000
answers_count = 50
#answers_count = 200
#max_len_words = 240
#max_len_words = 500

#'''
max_len_words = 800
max_len_code_char = 18
#'''

'''
max_len_words = 1000
max_len_code_char = 50
'''

#max_len_words = 1000
repr_flag = True

def minify_python(filename):
    mini_kwargs = {'rename': True, 'preserve': None, 'indent': 1, 'selftest': True, 'joinlines': True, 'docstrings': True, 'debug': False}
    return minipy.minify(filename, **mini_kwargs)

'''NEED TO ADD SOMETHING TO REMOVE DUPLICATES THAT MINIPY CAUSES'''
'''NEED TO LOWERCASE DESCRIPTIONS AND REMOVE ONE WORD STRINGS "capital" and "uppercase" IN THEM; OR JUST INIT WITH GloVe'''
'''NEED TO REMOVE CODEFORCES FOLDERS WITH NUMBER AT END OF FOLDER NAME; OR JUST ALTER DROPBOX LINK'''
'''ALSO, GET ANNOTATED DESCRIPTIONS LOADED'''

def load_from_dir(root_dir, main_count, sub_count, questions_count, max_len_words, repr_flag):
    for subdir, dirs, files in os.walk(root_dir):
        #print "dirs"
        #print subdir
        #print dirs
        #print files
        #asdf
        for file in files:
            if str(file) != "description_annotated.txt":
            #if str(file) != "description.txt":
                if os.path.join(subdir).split('/')[-1] == 'solutions_c++':
                    continue
                elif os.path.join(subdir).split('/')[-1] == 'description':
                #elif os.path.join(subdir).split('/')[-1] == 'description_annotated':
                    # inside_files = os.walk("/".join(subdir.split('/')[:-1])+ '/solutions_python').next()[0]
                    # print "inside_files count",inside_files,len(inside_files)
                    for s_subdir, s_dirs, inside_files in os.walk("/".join(subdir.split('/')[:-1])+ '/solutions_python'):
                        break
                    if main_count < questions_count:
                        main_count += 1
                        sub_count = 0
                        '''
                        if answers_count > len(inside_files):
                            main_count -= 1
                            break
                            '''
                        # else:
                        print("before opening file","main_count >>",main_count,"files count inside >>",len(inside_files))
                        with open(os.path.join(subdir, file), 'r') as myfile:
                            print "myfile"
                            print myfile
                            #some if statement need to go near here
                            #data=myfile.read().replace('\r', ' ').replace('\t', ' ').replace('\n', ' ').replace('.', ' ')
                            data=myfile.read()
                        if len(data.split(" ")) > max_len_words:
                            print("skipping short answer","main_count >>",main_count)
                            main_count -= 1
                            break
                        else:
                            # with open("questions.txt", "a") as fileappend:
                                # fileappend.write(data +".\n")
                            # look for files in .../101_E/solutions_python
                            for sub_subdir, sub_dirs, sub_files in os.walk("/".join(subdir.split('/')[:-1])+ '/solutions_python'):
                                for sub_file in sub_files: # look for files in .../101_E/solutions_python each answer

                                    if sub_count < answers_count:
                                        sub_count += 1
                                        '''
                                        with open(os.path.join(sub_subdir, sub_file), 'r') as myfile:
                                                sub_data=myfile.read()
                                                '''
                                        try:
                                            #sub_data=str(list(minify_python(os.path.join(sub_subdir, sub_file))))
                                            #sub_data=str(tokenize_python_code(minify_python(os.path.join(sub_subdir, sub_file))))
                                            sub_data=tokenize_python_code(minify_python(os.path.join(sub_subdir, sub_file)))
                                            # Comment this line below out:
                                            # to create 1 question to 1 answer.
                                            # Otherwise, it will repeat number of answers_count for Questions
                                            # to map to answers.
                                            if "#include" not in str(sub_data) and "using namespace std" not in str(sub_data) and len(sub_data) < max_len_code_char and len(sub_data) > 1:
                                                #if repr_flag == True:
                                                if "capital" not in str(data) and "uppercase" not in str(data):
                                                    '''NEED TO DELETE OLD train.* WHEN USING THIS'''
                                                    tmp_path=os.path.abspath(os.path.join('.', os.curdir)) + '/tmp'
                                                    with open(tmp_path + '/' + "train.questions", "a") as fileappend:
                                                        #data = data.replace('\r', '\\r').replace('\t', '\\t').replace('\n', ' \\n ').replace('\\n  \\n', '\\n \\n').replace('\\n  \\n', '\\n \\n').lstrip(' \\n').lstrip('\\n').rstrip(' \\n').rstrip(' ').rstrip('\\n')
                                                        #data = data.replace('\r\n', '\n').replace('\n', ' ¶ ').replace('¶  ¶', '¶ ¶').rstrip(' ¶').rstrip(' ').rstrip('¶').replace('† ', '† ').replace(' ‡', ' ‡')
                                                        #data = data.replace('\r', ' ').replace('\t', ' ').replace('\n', ' ').replace('.', ' ')

                                                        '''STILL NEED TO MAKE QUESTIONS LOWERCASE INSIDE OF question_to_fields_final_perfect.'''
                                                        '''NEED TO DO VIA .decode(UTF-8) says stackoverflow.com/questions/6797984/how-to-convert-string-to-lowercase-in-python'''
                                                        data_split = str(question_to_tokenized_fields(data.replace('\r\n', '\n').replace('\n', ' ¶ ').replace('¶  ¶', '¶ ¶').rstrip(' ¶').rstrip(' ').rstrip('¶').replace('† ', '† ').replace(' ‡', ' ‡').replace('Link to Russian translation of problem', ' ').replace('Link to Russian translation the problem', ' ').replace('The link to the Russian translation.', ' ').replace('See Russian Translation', ' ').replace('View Russian Translation', ' ').replace('Russian Translation', ' ')))

                                                        fileappend.write(data_split +"\n")
                                                        #fileappend.write(data +".\n")
                                                    with open(tmp_path + '/' + "train.answers", "a") as sub_fileappend:
                                                        #sub_fileappend.write(repr(sub_data)[1:-1] + "\n")
                                                        #sub_data = sub_data.replace('\r', ' ').replace('\t', ' ').replace('\n', ' ').replace('.', ' ')
                                                        #sub_fileappend.write(sub_data + " ¿\n")
                                                        sub_fileappend.write(str(sub_data) + "\n")

                                                #elif repr_flag == False:
                                                    '''
                                                    with open("questions.txt", "a") as fileappend:
                                                        fileappend.write(data +".\n")
                                                    with open("answers.txt", "a") as sub_fileappend:
                                                        sub_fileappend.write(sub_data + "\n")
                                                        '''
                                                    # sub_fileappend.write(os.path.join(sub_subdir, sub_file)+".\n")
                                                print "main_count >>",main_count,"sub_count",sub_count, len(sub_files)
                                        except SyntaxError:
                                            pass

                                        else:
                                            pass
                                    else:
                                        break

                    else:
                        break

load_from_dir(rootdir1, main_count, sub_count, questions_count, max_len_words, repr_flag)
load_from_dir(rootdir2, main_count, sub_count, questions_count, max_len_words, repr_flag)
load_from_dir(rootdir3, main_count, sub_count, questions_count, max_len_words, repr_flag)



