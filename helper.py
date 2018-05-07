# FUNCTIONS FOR CALCULATING METRICS OF STRINGS
import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import networkx as nx


max_substring_len = 4


# AUXILLARY FUNCTIONS
def get_alphabet(T):
    sigma = []
    for example in T:
        assert type(example) == list
        for token in example:
            assert type(token) == tuple
            assert len(token) == 1
            if token[0] not in sigma:
                sigma.append(token[0])
    return set(sigma)


def get_all_substrings(T,mult=1):
    # returns a list of sequencese
    substrings = []
    for w in T:
        for k in range(1, mult*max_substring_len+1):
            for i in range(len(w)-k+1):
                sequence = make_new_sequence(w[i:i+k])
                substrings.append(sequence)
    return substrings


def generate_new_NT(G):
    return "NT"+str(len(G[1].union(G[-1])))


# given a list representing nodes in a connected component
# returns False if there is less than or more than 1 non-terminal present
# otherwise it returns the nonterminal
def single_NT(component):
    nt_present = 0
    for sequence in component:
        assert type(sequence) == tuple
        if len(sequence)==1 and sequence[0][:2]=="NT":
            alpha = sequence
            nt_present += 1
    if nt_present == 1:
        return alpha
    else:
        return False


def token_in_sequence(token, sequence):
    for w in sequence:
        for i in range(len(w)):
            for k in range(1, max_substring_len+1):
                if make_new_token(sequence[i:i+k]) == token:
                    return True
    return False


def sequence_in_T(seq, T):
    # check if a sequence is within the given examples
    assert type(seq) == tuple
    assert type(seq[0]) == str
    assert type(T) == list 
    for example in T:
        string = ""
        for token in example:
            if type(token[0]) != str:
                print(token[0])
            string+=token[0]
        if "".join(seq) in string:
            return True
    return False


def make_new_sequence(lst):
    # a sequence is a tuple of length >=1
    # make "" the epsilon character
    new_lst = []
    for element in lst:
        if type(element) != list and type(element) != tuple:
            if element != "":
                new_lst.append(element)
        else:
            for subelement in element:
                if subelement != "":
                    new_lst.append(subelement)
    return tuple(new_lst)

