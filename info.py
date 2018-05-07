# Calculating the information Theoretic Stuff
# FUNCTIONS FOR CALCULATING METRICS OF STRINGS
import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import networkx as nx

from helper import *

max_substring_len = 4


# get probability of all substrings returned as counter object
# the keys are tuples of tokens, which represents the substring
def get_probabilities(T):
    """
    T: List of examples. An example is a list of tokens, does not include # symbol
    ret: Counter object. key is a sequence, value is the probability of sequence occuring in T
    """
    prob = Counter()

    # calculate the denominators to use
    maxp2 = 3*max_substring_len
    Nks = [ 0 for i in range(maxp2)]
    for example in T:
        assert ("#",) not in example
        for k in range(1, maxp2+1):
            if k <= len(example):
                Nks[k-1] += len(example) - k + 1

    # calculate the probabilities
    for example in T:
        # need to add two to substring length here too
        for k in range(1, maxp2+1):
            for i in range(len(example)-k+1):
                w = make_new_sequence(example[i:i+k])
                if len(example) >= k:
                    #prob[w] += 1./(N-k+1)
                    prob[w] += 1./Nks[k-1] # assume strings overlap
                    #prob[w] += k/Nks[k-1] # assume strings cannot overlap
                else:
                    #print(w)
                    prob[w] += 0
    total = sum(prob.values())
    for elem, count in prob.items():
        prob[elem] = count/total

    return prob


# get the expected value of all substrings of length
def get_expected_values(G, T):
    """
    T: List of examples. An example is a list of tokens
    ret: Counter object. key is a sequence, value is the expected value of a sequence occuring in T
    """
    Ex = Counter()
    # having mult == 1 was causing mutual information to be negative
    all_substrings = get_all_substrings(T, mult=3)
    prob = get_probabilities(T)

    #sigma = G[0].union(set([""]))
    sigma = all_substrings + [("",)] # should be this, but expansive
    sigma = list(filter(lambda x: len(x)<3, sigma))
    #for terminal in sigma:
    #    assert type(terminal) == str
    for example in T:
        ex = [("#",)] + example + [("#",)]
        #ex = example
        # make the length of looked at substrings + 2 higher
        for k in range(1, 3*max_substring_len+1):
            for i in range(len(ex)-k+1):
                w = ex[i:i+k]
                #print(w, w[0], w[-1])
                w_token = make_new_sequence(w)
                if w_token not in Ex.keys():
                    # this assumes that epsilon is a valid character
                    Ew = 0
                    if (w[0] == ("#",)) and (w[-1] == ("#",)):
                        win = make_new_sequence(w[1:-1])
                        Ew = prob[win]
                    elif (w[0] == ("#",)) and (w[-1] != ("#",)):
                        for v in sigma:
                            wv = make_new_sequence(w[1:]+[v])
                            if wv in all_substrings:
                                #print("wv", wv, w, Ew)

                                Ew += prob[wv]
                    elif (w[0] != ("#",)) and (w[-1] == ("#",)):
                        for u in sigma:
                            uw = make_new_sequence([u]+w[:-1])
                            if uw in all_substrings:
                                #print("uw", uw, w, Ew)
                                Ew += prob[uw]
                    else:
                        for u in sigma:
                            for v in sigma:
                                uwv= make_new_sequence([u]+w+[v])
                                if uwv in all_substrings:
                                    #print("uwv", uwv, w, Ew)
                                    Ew += prob[uwv]
                    # wasn't adding this, just assigning.
                    # still doesnt fix negative mutual information problem
                    Ex[w_token] = Ew
    #print(Ex)
    assert type(list(Ex.keys())[0]) == tuple
    return Ex


def get_freq(T):
    freq = Counter()
    for example in T:
        for k in range(1, max_substring_len+1):
            for i in range(len(example)-k+1):
                sequence = make_new_sequence(example[i:i+k])
                freq[sequence] += 1
    return freq


def get_mutual_information(substrings, G, T):
    # substrings are sequences, meaning tuples. They are the only
    # substrings that we want to get info on
    MI = Counter()
    assert type(substrings) == list
    assert type(substrings[0]) == tuple
    if len(T) > 0:
        assert ("#",) not in T[0]
    Tp = []
    for example in T:
        Tp.append([("#",)] + example + [("#",)])


    E = get_expected_values(G, T) # of all substrings of length k
    #print(E)
    # make it so that "#" character is in examples

    sigma = G[0]
    for sequence in set(substrings):
        if sequence not in MI.keys():
            mi = 0
            for l in sigma.union(set(["#"])):
                for r in sigma.union(set(["#"])):
                    lw  = make_new_sequence([l, sequence   ])
                    if sequence_in_T(lw, Tp):
                        wr  = make_new_sequence([   sequence, r])
                        if sequence_in_T(wr, Tp):
                            lwr = make_new_sequence([l, sequence, r])
                            if sequence_in_T(lwr, Tp):
                                Elwr, Ew = E[lwr], E[sequence]
                                Elw, Ewr = E[lw], E[wr]
                                if Elwr != 0 and Elw != 0 and Ewr != 0 and Ew != 0:
                                    lg = np.log((Elwr*Ew)/(Elw*Ewr))
                                    if lg < -10:
                                        #print("in seq, but not in E??")
                                        print("negative MI")
                                        print(sequence, Ew)
                                        print(lwr, Elwr)
                                        print(wr, Ewr)
                                        print(lw, Elw)
                                        #print(l, sequence, r)
                                        #print(Ew, Elw, Ewr, Elwr)
                                        #print(Elwr*Ew, Elw*Ewr, lg)
                                        print()
                                    mi += (Elwr/Ew)*lg
                                else:
                                    # only happens if str too long
                                    pass
                                    #print("in seq, but not in E??")
                                    #print(sequence, Ew)
                                    #print(lwr, Elwr)
                                    #print(wr, Ewr)
                                    #print(lw, Elw)
            MI[sequence] = mi
    return MI



# creates a new list of frequently occuring substrings
# where a substring is a tuple of tokens
def prune_substrings(substrings, G, T):
    # length 1 substrints have an extra comma for some reason
    #assert type(substrings) == type([])
    #assert type(substrings[0]) == tuple

    new_substrings = []
    freq = get_freq(T)
    N = len(freq)
    for sequence, count in freq.most_common():
        if count > 0:#len(T)//3:
            new_substrings.append(sequence)

    m_info = get_mutual_information(new_substrings, G, T)
    new_substrings = []
    print(m_info)
    N = len(m_info)
    #print(m_info)
    for sequence, count in m_info.most_common():
        #print(sequence, len(sequence), count)
        if abs(count) > 0:
            new_substrings.append(sequence)

    """
    print("Frequency")
    print(freq.most_common())
    print()
    print("Mutual Information")
    print(m_info.most_common())"""
    return new_substrings



# TALK ABOUT THE FOLLOWING DRAWBACKS OF THE ALGORITHM
# few drawbacks. very inefficient.
# they stated briefly can take anywhere from a few minutes to a few hours o infer from one set of examples
# the algorithm is O(n^2 v^2 |T|^2), which is large
# it is also dependent on certain hyperparameters chosen, which makes it very finicky.
# they also did not publish the hyperparameters he used, Though the success of them is going
# to differ based on the size of examples, the number of terminals, etc..


