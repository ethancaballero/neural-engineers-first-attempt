# Neural Engineers TF

NE implementation in TensorFlow for learning to engineer.

This model is a [decription2code task](https://github.com/openai/requests-for-research/pull/5) baseline (rnn-based seq2seq with AST decoder & adaptive number of attention hops per decode step) (with two stage SL then RL training) that yielded underwhelming results.

## Acknowledgments
[Barronalex](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow)'s and [Therne](https://github.com/therne/dmn-tensorflow)'s tf-DMN+s,

[Alexander Johansen's seq2seq modules](https://github.com/alrojo/tensorflow-tutorial/blob/master/lab3_RNN/tf_utils.py#L11),

Jon Gauthier's [RML attempt](https://github.com/hans/ipython-notebooks/blob/master/Reward-augmented%20maximum%20likelihood%20learning%20for%20autoencoders.ipynb),

MarkNeumann's [adapative attention module](https://github.com/DeNeutoy/act-rte-inference/blob/master/AdaptiveIAAModel.py#L197-L247)

and seqGANs [policy gradient modules](https://github.com/LantaoYu/SeqGAN/tree/master/pg_bleu).


## Repository Contents
| file | description |
| --- | --- |
| `dmn_plus.py` | contains the DMN+ model |
| `dmn_train.py` | trains the model on a specified (-b) babi task|
| `dmn_test.py` | tests the model on a specified (-b) babi task |
| `get_data.sh` | shell script to fetch & load Description2Code Dataset |


-----

##### Instructions to get data and run model: (Work In Progress)


1. The full problems dataset can be downloaded and pre-processed can be generated through:

```
  sh get_data.sh
```

To tweak for new directories:

2. The root directories can be set for the coding problems in `yads_data_loader.py` :

```
  rootdir1 = './description2code_current/codeforces_delete'
  rootdir2 = './description2code_current/hackerearth/problems_college'
  rootdir3 = './description2code_current/hackerearth/problems_normal'
```

3. Then tweak these parameters per needed experiment:

```
questions_count = 3000
answers_count = 50
max_len_words = 800
regular_desc = True
```
4. to run the model:

```
 python dmn_train.py
```

###### Note about the data/sample:
----

There's some toy data in tmp folder for if you don't feel like downloading full data.
If you plan on using full data, delete all files in tmp folder before running `sh get_data.sh` and maybe before `python dmn_train.py` as well.


## Results:
outputs always overfit to simplistic incorrect answers such as 
`a=raw_input(); print(2-a)`

Interestingly, model learned dynamic adaptation of number attention hops (per each decode step) that seems to correlate with complexity of each decode step:

values in 1st line are number of attention hops for each decode step;
values in 2nd line are output of each decode step.

The two lines are aligned below for visualization purposes:
```
num of attn hops per dec step: array([ 1., 1., 6.,          1., 2., 1., 3.,       6., 2., 7.,  2.,  3.])
output of dec step:                    a   =   raw_input(       )   ;   print(    2   -   a     )   END
```


## How I would re-do project today:
-Transformer from "attention is all you need" https://arxiv.org/abs/1706.03762

-Use a Differentiable Programming Language as decribed in "Differentiable Functional Program Interpreters" https://arxiv.org/abs/1611.01988 so that you can scrap all score/value estimators from RL and just get the exact gradient of the reward with respect to the parameters

-Hindsight Experience Replay https://arxiv.org/abs/1707.01495

-unsupervisedly pretrain the encoder & decoder

-better/bigger data such as maybe the 150370 docstring2code pairs from https://github.com/EdinburghNLP/code-docstring-corpus or the 17000 programming challenge description2code pairs (mine was just 7000) that Alex Skidanov recently acquired.


^if you're interested in trying any of this, message me or join [near.ai](http://near.ai/)




