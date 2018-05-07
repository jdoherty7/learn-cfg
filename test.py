# Testing functions
import numpy as np
from collections import defaultdict
import networkx as nx

import nltk
from nltk.parse import ShiftReduceParser
from nltk.parse.generate import generate


from helper import *
from main import *
from check import *
from info import *

# T is a list containing lists of tokens
def get_T(name="a confused mind.txt"):
    with open(name, 'r') as f:
        doc_list = f.readlines()
    T = []
    for i in range(len(doc_list)):
        s = doc_list[i].replace('\n', '').lower()
        if s is not "":
            T.append((s,))
    return T


def test_T():
    # An example is a list of tokens
    T = get_T()
    assert type(T) == list
    for example in T:
        assert type(example) == list
        # a token is a length 1 tuple
        for token in example:
            assert type(token) == tuple
            assert len(token) == 1


def test_probabilities():
    T = get_T()
    probs = get_probabilities(T)
    print("get_probabilities() doesn't fail.")


def test_freq():
    T = ["hi", "mi", "doggo"]
    T = [list(map(lambda x: (x,), example)) for example in T]
    probs = get_probabilities(T)

    freq = get_freq(T)
    assert freq[("i",)] == 2
    assert freq[("y",)] == 0
    assert freq[("h", "i")] == 1
    assert freq[("o",)] == 2
    assert freq[("#",)] == 0
    T = get_T()
    freq = get_freq(T)
    print("get_freq() function works()")


def test_expected_values():
    pass

def test_mutual_information():
    T = [["the cat is"],
         ["the mat is"],
         ["the what is"],
         ["how is"]]

    T = list(map(lambda w: list(map(lambda t: (t,), w[0])), 
            T))
    G = get_alphabet(T), set(), {}, set()
    substrings = get_all_substrings(T)

    minfo = get_mutual_information(substrings,G,T)
    # mutual information should be non-negative
    for elem, count in minfo.items():
        assert count >= 0


def test_sequence_in_T():
    T = get_T()
    T = T[:len(T)//3]
    substrings = get_all_substrings(T)
    # test sequence_in_T
    for sequence in substrings:
        assert sequence_in_T(sequence, T)





# test each character in set of symbols, including spaces
def test_get_alphabet():
    # a terminal symbol is a string
    T = ["hi", "my", "dog"]
    T = [list(map(lambda x: (x,), example)) for example in T]

    sigma = get_alphabet(T)
    actual = set("h i m y d o g".split())
    assert sigma == actual
    assert type(sigma) == set
    for terminal in sigma:
        assert type(terminal) == str

    T = ["hi ", "my", "dog"]
    T = [list(map(lambda x: (x,), example)) for example in T]

    sigma = get_alphabet(T)
    actual = set("h i m y d o g".split()).union(set([" "]))
    assert type(sigma) == type(actual)
    assert sigma == actual
    print("get_alphabet() function works.")



def test_get_all_substrings():
    print(get_all_substrings([]))
    assert list     == type(get_all_substrings([]))
    assert []       == get_all_substrings([])
    assert [('a',)] == get_all_substrings([('a',)])

    T = ["hi", "dog"]
    T = [list(map(lambda x: (x,), example)) for example in T]
    substrings = get_all_substrings(T)
    actual = [("h",), ("d", "o", "g"),
              ("i",), ("d", "o"), ("o", "g"),
              ("h", "i"), ("d",), ("o",), ("g",)]
    assert len(actual) == len(substrings)
    for token in actual:
        assert token in substrings
    print("get_all_substrings() function works")



def test_single_NT():
    # each node in a substitution graph is a sequence
    # Each component is a list of nodes, where a node is a sequence
    c1 = [("NT0",), ("a", 'd'),   ("b",)]
    c2 = [("a",),  ("c",),   ("NT37",), ("d",)]
    c3 = [("a",'NT5'),  ("c",),   ("nt",),   ("d",)]
    c4 = [("a",'m'),  ("NT0",), ("NT37","NT3"), ("d",)]
    c5 = [("a",'b'),  ("NT1","NT2"),   ("c","l", 'e', 'r'), ("d",)]
    c6 = [("a",'b'),  ("N","m"),   ("c","l", 'e', 'r'), ("d",)]
    c7 = []
    components = [c1, c2, c3, c4, c5, c6, c7]
    # Should number three be NT5 or False? i think it should be nt5..
    boolean    = ["NT0", "NT37", "NT5", False, False, False, False]
    for i in range(len(components)):
        component = components[i]
        assert single_NT(component) == boolean[i]
    print("single_NT() function works.")



def test_generate_new_NT():
    nt = generate_new_NT((set([]),set([]),set([]),set([])))
    assert nt == "NT0"
    nt = generate_new_NT((set([]),set([1]),set([]),set([])))
    assert nt == "NT1"
    nt = generate_new_NT((set([]),set([]),set([]),set([3])))
    assert nt == "NT1"
    nt = generate_new_NT((set([]),set([7]),set([]),set([3])))
    assert nt == "NT2"
    print("generate_new_NT() function works")



def test_sg():
    T = [" This ", " That ", "I am a man", "I am a woman"]
    T = [list(map(lambda x: (x,), example)) for example in T]
    LearnCFG(T)
    sigma = get_alphabet(T)
    G = (sigma, set([]), set([]), set([]))
    SG = construct_substitution_graph(G, T, prune=False)
    print(SG.nodes())
    #for component in nx.connected_components(SG):
    #    print(component)
    assert ("T","h","i","s") in SG.nodes()
    assert ("T","h","a","t") in SG.nodes()
    assert ("a","m") in SG.nodes()
    assert ("T","h","i","s", " ") in SG.nodes()

    #T = get_T()
    #sigma = get_alphabet(T)
    #G = (sigma, set([]), set([]), set([]))
    #SG = construct_substitution_graph(G, T, prune = True)
    #SG = prune(SG, G, T)
    #print(SG.nodes())
    #print(list(nx.connected_components(SG)))
    #for cc in nx.connected_components(SG):
        #print(cc)
    print("construct_substitution_graph() function works")



def test_prune():
    pass



def test_check():
    productions1 =  """
                    NT0 -> 'a' NT1
                    NT0 -> 'b' NT1
                    NT1 -> 'c' NT1
                    NT1 -> 'm'
                    """.strip()
    test_grammar = nltk.CFG.fromstring(productions1)
    s2t = lambda string: list(map(lambda x: (x,), string))

    print(s2t("accccccccm"))

    terminals = set(["a", "b", "c", "m"])
    nonterminals = set(["NT0", "NT1"])
    productions = {"NT0": [[("a",), ("NT1",)], [("b",), ("NT1",)]],
                   "NT1": [[("c",), ("NT1",)], [("m",)]]}
    start = set(["NT0"])
    G = (terminals, nonterminals, productions, start)
    #for sentence in generate(convert2_nltk_CFG(G), n=10):
    assert check(test_grammar, s2t("accccccccm"), nltk=True)
    assert check(test_grammar, s2t("bcccccccm"), nltk=True)
    assert check(test_grammar, s2t("am"), nltk=True)
    assert check(test_grammar, s2t("acccccdccm"), nltk=True) == False
    assert check(test_grammar, s2t("bccccmc"), nltk=True) == False
    assert check(test_grammar, s2t("am!"), nltk=True) == False

    print("check() function works for nltk grammars")
    # check that convert2_nltk_CFG function works
    assert check(G, s2t("accccccm"))
    assert check(G, s2t("bm"))
    assert check(G, s2t("bccccccm"))
    assert check(G, s2t("acccccmb")) == False
    assert check(G, s2t("bmb!")) == False
    assert check(G, s2t("bcNT0ccccc3m")) == False
    print("check() function works for this code bases' grammars")
    print("convert2_nltk_CFG() function works.")


def test_random_grammars():
    for _ in range(50):
        G = generate_random_grammar()
        T = generate_positive_examples(G)
        nltk_grammar = convert2_nltk_CFG(G)
        #print(nltk_grammar)
        #print(T)
        for w in T:
            #print(w)
            assert check(nltk_grammar, w, nltk=True)
            assert check(G, w)
    print("generate_random_grammar() and generate_positive_examples() functions work")


def test_alpha():
    sigma = set(['a', 'b', 'c'])
    NTs = set(['NT0', 'NT1'])
    productions = { "NT0": [[('a',), ('b',), ('NT1',)],
                            [('b',), ('a',)]],
                    "NT1": [[('a',), ('c',)],
                            [('b',), ('NT1',)]]}
    start = set(["NT0"])
    G = (sigma, NTs, productions, start)
    # should be able to handle strings that are
    # not actually in the given grammar. returns w
    T = [[('b',), ('a',)],
         [('a',), ('b',), ('b',), ('b',)]]
    Tred = [[('NT0',)],
            [('a',), ('b',), ('b',), ('b',)]]
    Tp = []
    for w, wr in zip(T, Tred):
        #print(wr)
        #print(alpha(w, G))
        #print()
        assert wr == alpha(w, G)



def run_tests():
    test_alpha()
    test_check()
    test_random_grammars()
"""
    test_generate_new_NT()
    test_get_all_substrings()
    #test_sg()
    test_probabilities()
    test_freq()
    test_get_alphabet()
"""

if __name__ == "__main__":
    #test_freq()
    #test_mutual_information()
    test_sg()
    #run_tests()

"""
The parser was implemented using the Natural Language Toolkit.

Grammars are defined as a set of terminal symbols, a set of nonterminal symbols,
a dictionary of productions with a non terminal symbol as the key and a list/set?
of transitions. A transition is a list of tokens (terminal and nonterminal symbols)
The last part of the grammar is a set of start symbols. This should just
be a set of one, the non terminal symbol ("NT0"). Is this symbol included
in the nonterminal set as well?? I don't think so?


"""