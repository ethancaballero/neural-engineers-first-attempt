perl tokenizer.perl -l 'en' < validate.answers > validate.answers.tok

perl tokenizer.perl -l 'en' < validate.questions > validate.questions.tok

perl tokenizer.perl -l 'en' < train.answers > train.answers.tok

perl tokenizer.perl -l 'en' < train.questions > train.questions.tok

python build_dictionary.py train.answers.tok

python build_dictionary.py train.questions.tok

python shuffle.py train.questions.tok train.answers.tok