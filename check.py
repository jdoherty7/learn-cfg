#Create and Check Grammars
import numpy as np
import random
from copy import copy

import nltk
from nltk.parse import ShiftReduceParser
from nltk.parse.generate import generate

from helper import *
from info import *



# assume non_terminal is a string
# this returns all rules as a list of lists.
# inner list is a squence of tokens representing the rule
def get_production_rules(G, non_terminal):
    return G[2][non_terminal]


# Returns a random CFG grammar nltk object and a list of sequences
# of tokens in the language
def generate_random_grammar():
    alphabet = list(map(chr, range(97,123)))
    sigma, N, P, S = [], [], {}, ["NT0"]
    # Make a random alphabet, set of terminals
    x = np.arange(26)
    np.random.shuffle(x)
    rs = x[:random.randint(3,25)]
    for i in rs:
        sigma.append(alphabet[i])

    # make a random set of Non terminals, dont include 0 in N
    # right now N includes all non terminals, including start
    N = list(map(lambda x: "NT"+str(x), range(0, random.randint(3, 6))))

    # make a random number of rules for each non terminal. Add each new rule to the list
    all_symbols = alphabet + N
    for nt in set(N):
        P[nt] = []
        num_rules = random.randint(1,8)
        for i in range(num_rules):
            new_rule = []
            for i in range(random.randint(2,4)):
                pick = random.randint(1, len(all_symbols)-1)
                # value is a list of tokens
                new_rule.append((all_symbols[pick],))
            P[nt].append(new_rule)

    assert type(P) == dict
    assert type(list(P.keys())[0]) == str
    assert type(list(P.values())[0]) == list
    assert type(list(P.values())[0][0]) == list
    assert type(list(P.values())[0][0][0]) == tuple
    assert type(sigma[0]) == str
    assert type(N[0]) == str
    assert type(S[0]) == str

    G = (set(sigma), set(N), P, set(S))
    return G

def generate_handmade_grammar():
    G0 = (set(["a", "b"]),
          set(["NT0", "NT1"]),
          {"NT0": [[("a",), ("NT0",)], 
                   [("NT1",)]],
           "NT1": [[("b",)]]},
          set(["NT0"])
        )
    G1 = (set(["a", "b", "c"]),
          set(["NT0", "NT1"]),
          {"NT0": [[("a",), ("NT0",)], 
                   [("c",), ("NT1",)]],
           "NT1": [[("b",), ("c",)]]},
          set(["NT0"])
        )
    return G1

def generate_python():
    T = [   
        ["a=4"],
        ["b=7"],
        ["c=a*b"],
        ["print(c)"],
        ["def f():\n\tprint('hi')"],
        ["def g():\n\treturn 5"],
        ["if c==4:\n\tc=c+3\nelse:\n\tprint(b)"],
        ["if a>4:\n\ta=2\nelse:\n\tc=3"]
        ]
    Tp = []
    for example in T:
        Tp.append(list(map(lambda x: (x,), example[0])))
    return Tp

def generate_haskell():
    T = [   
        ["let a=3 in a*5"],
        ["if a==5 then b else (a+7)"],
        ["let d=7 in d+5"],
        ["if d>7 then 4 else (a-7)"],
        ["let m=7 in v*8+1"],
        ["fac 0 = 1\nfac n = n*(fac (n-1))"],
        ["inc x = x+1"],
        ["f 0 = 0\nf a = f (a-1)"]
        ]
    Tp = []
    for example in T:
        Tp.append(list(map(lambda x: (x,), example[0])))
    return Tp

# returns a list of sequences of tokens
# sequence is a list and tokens are strings
def generate_positive_examples(G, N=None):
    if N is None:
        N = random.randint(3, 10)
    T = []
    nltk_grammar = convert2_nltk_CFG(G)
    # If grammer has no ending then it will be infinite
    # Really no grammar like this should be generated but
    # really limiting the depth is fine to stop it but the
    # technically the examples wont be in G

    # sentence is originally a list of terminal symbols
    # change it to a list of tokens
    # this generates examples in order
    # depth is the max sentence size desired
    d = min(np.log2(len(G[0].union(G[1]))**2), 8)
    for sentence in generate(nltk_grammar, n=50*N, depth=15):
        #print(sentence)
        #print(len(sentence))
        tokens = list(map(lambda x: (x,), sentence))
        if check(nltk_grammar, tokens, nltk=True):
            T.append(tokens)
    #print("don")
    # randomize the order of the strings
    random.shuffle(T)
    #print("len T", len(T))
    #print("lenT[N]", len(T[:N]))
    TN = []
    Nm = min(len(T), N)
    for i in range(Nm):
        TN.append(T[i])
    #print("l TN", len(TN))
    return TN


# output type is a tuple to be put in Production()
# the output is a sequence, that is a tuple of terminals/nonterminals
# input is a rule, which is a list of tokens
def rule_to_tuple(rule, nonterminals):
    rhs = []
    for token in rule:
        assert type(token) == tuple
        assert len(token) == 1
        symbol = token[0]
        if symbol in nonterminals:
            rhs.append(nltk.Nonterminal(symbol))
        else:
            rhs.append(symbol)
    return tuple(rhs)



# nltk.grammar.CFG(start, productions)
# terminals and nonterminals are implicitly defined in productions
# http://www.nltk.org/_modules/nltk/grammar.html#CFG
# http://www.nltk.org/_modules/nltk/grammar.html#Production
def convert2_nltk_CFG(G):
    terminals, NTs, P, S = G
    Prod = copy(P)
    # this is here to ensure full coverage of terminals
    # when parsing the grammar for testing
    Prod["DUMMY"] = [list(map(lambda x:(x,), terminals))]
    assert len(S) > 0 # need a start symbol
    if len(S) > 1:
        if "NT0" not in Prod.keys():
            Prod["NT0"] = []
        for Si in S:
            Prod["NT0"].append([(Si,)])
    assert "NT0" in S
    start = nltk.Nonterminal("NT0")
    nltk_nts = nltk.nonterminals(" ".join(list(NTs)))
    productions = []
    # only look at nonterminals with productions
    for NT in Prod.keys():
        for rule in Prod[NT]:
            rhs = rule_to_tuple(rule, NTs)
            #print("convert", NT, rhs)
            prod = nltk.Production(nltk.Nonterminal(NT), rhs)
            productions.append(prod)
    # production is empty here...
    return nltk.grammar.CFG(start, productions)



# checks if the language defining the grammar accepts the given string.
# set nltk if G is already an nltk grammar object
# this sometimes fails but it fails because the string is not actually
# in the language which is caused by restricting the length of the string
def check(G, tokens, nltk=False):
    assert type(tokens) == list
    # convert list of tuples to list of strings
    if len(tokens) > 0:
        if type(tokens[0]) == tuple:
            tokens = [token[0] for token in tokens]
        assert type(tokens[0]) == str
    if not nltk:
        _, _, P, _ = G
        if len(P) == 0:
            return False
        grammar = convert2_nltk_CFG(G)
    else:
        grammar = G
    sr = ShiftReduceParser(grammar)
    #print(grammar.productions())

    # parse requires a series of tokens
    try:
        # this will raise an exception if fails
        # check if all tokens are terminals in grammar
        grammar.check_coverage(tokens)
    except Exception as e:
        return False

    # check that token sequence has some parse tree
    #print(list(sr.parse(tokens)))
    if len(list(sr.parse(tokens))) > 0:
        return True
    else:
        return False



# CALCULATING ACCURACY AND CORRECTNESS OF SOLUTION

def get_correct(grammar, examples):
    grammars_correct = 0
    examples_correct = 0

    b = True
    for w in examples:
        if not check(grammar, w):
            b = False
        else:
            examples_correct += 1
    if b:
        grammars_correct += 1    
    return grammars_correct, examples_correct


def get_acc(errors, N):
    N = len(errors)
    if N == 0:
        return 0, 0
    ga, ea = 0., 0.
    for gr, ex in errors:
        ga += gr
        ea += ex
    return ga/len(errors), ea/N

