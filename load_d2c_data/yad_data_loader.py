# -*- coding: utf-8 -*-




import os
import minipy
from tokenize_python import tokenize_python_code
from question_to_fields_final_perfect import question_to_tokenized_fields


rootdir1 = './description2code_current/codeforces_delete'
rootdir2 = './description2code_current/hackerearth/problems_college'
rootdir3 = './description2code_current/hackerearth/problems_normal'


main_count = 0
sub_count = 0
questions_count = 3000
answers_count = 50

max_len_words = 800

# true means pick up the normal descriptions without
# weird cross signs.
# false means: "it basically gives the attentions supervision" - EC
regular_desc = True

def minify_python(filename):
    mini_kwargs = {'rename': True, 'preserve': None, 'indent': 1, 'selftest': True, 'joinlines': True, 'docstrings': True, 'debug': False}
    return minipy.minify(filename, **mini_kwargs)

def main_file_walker(root_dir, main_count, sub_count, questions_count, max_len_words,answers_count,regular_desc):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if regular_desc and str(file) != "description_annotated.txt":
                if os.path.join(subdir).split('/')[-1] == 'solutions_c++':
                    break
                elif os.path.join(subdir).split('/')[-1] == 'description':
                    # inside_files = os.walk("/".join(subdir.split('/')[:-1])+ '/solutions_python').next()[0]
                    # print "inside_files count",inside_files,len(inside_files)
                    for s_subdir, s_dirs, inside_files in os.walk("/".join(subdir.split('/')[:-1])+ '/solutions_python'):
                        break
                    if main_count < questions_count:
                        main_count += 1
                        sub_count = 0
                        if answers_count > len(inside_files):
                            main_count -= 1
                            break
                        # else:
                        print("before opening file","main_count >>",main_count,"files count inside >>",len(inside_files),"files name's:",file)
                        with open(os.path.join(subdir, file), 'r') as myfile:
                            problem_id = subdir.split('/')[- 2]
                            # data=myfile.read().replace('\r', ' ').replace('\t', ' ').replace('\n', ' ').replace('.', ' ')
                            data=myfile.read()
                            data = str(question_to_tokenized_fields(data.replace('\r\n', '\n').replace('\n', ' ¶ ').replace('¶  ¶', '¶ ¶').rstrip(' ¶').rstrip(' ').rstrip('¶').replace('† ', '† ').replace(' ‡', ' ‡').replace('Link to Russian translation of problem', ' ').replace('Link to Russian translation the problem', ' ').replace('The link to the Russian translation.', ' ').replace('See Russian Translation', ' ').replace('View Russian Translation', ' ').replace('Russian Translation', ' ')))

                            print(problem_id,"file_name")
                        # determine max-len to go through

                        if len(data.split(" ")) > max_len_words:
                            print("skipping short answer","main_count >>",main_count)
                            main_count -= 1
                            break
                        else:
                            with open("questions.txt", "a") as fileappend:
                                fileappend.write(data +"\n")
                                print(sub_count,"Subcount while adding questions")
                                # answers array [[ans1],[ans2]]
                                sub_data_array = []
                            # look for files in .../101_E/solutions_python
                            for sub_subdir, sub_dirs, sub_files in os.walk("/".join(subdir.split('/')[:-1])+ '/solutions_python'):
                                for sub_file in sub_files: # look for files in .../101_E/solutions_python each answer

                                    if sub_count < answers_count:
                                        sub_count += 1
                                        sub_data=tokenize_python_code(minify_python(os.path.join(sub_subdir, sub_file)))

                                        # Collect all Answers into 1 array to be aligned with answer
                                        sub_data_array.append(sub_data)


                                        # When the question count reaches the number we want
                                        # This will append All answers as 1 array to the answers file.
                                        if sub_count == answers_count:
                                            with open("answers.txt", "a") as sub_fileappend:
                                                sub_fileappend.write(str(sub_data_array) + "\n")
                                                # sub_fileappend.write(os.path.join(sub_subdir, sub_file)+".\n")
                                        print "main_count >>",main_count,"sub_count",sub_count, len(sub_files)
                                    else:
                                        break


            elif not regular_desc and str(file) == "description_annotated.txt":
                print("paste the above code here and it will pick annotated code")



main_file_walker(rootdir1, main_count, sub_count, questions_count, max_len_words,answers_count,regular_desc)
main_file_walker(rootdir2, main_count, sub_count, questions_count, max_len_words,answers_count,regular_desc)
main_file_walker(rootdir3, main_count, sub_count, questions_count, max_len_words,answers_count,regular_desc)
