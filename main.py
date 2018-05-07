
# Main functions
import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx

import re

from helper import *
from check import *
from info import *


# NTS = Non-terminally Separated
# NTS property guarantees that we have a Church Rosser System

max_substring_len = 4


#slow method
def construct_substitution_graph(G, T, prune=True):
    # Sequences are nodes
    sigma,_,_,_ = G
    SG = nx.Graph()
    Tp = []
    for example in T:
        Tp.append([("#",)] + example + [("#",)])

    # list of all sequences
    all_substrings = get_all_substrings(T)
    if prune:
        substrings = prune_substrings(all_substrings, G, T)
    else:
        substrings = all_substrings

    substrings = list(set(substrings))
    all_substrings = get_all_substrings(Tp)
    all_substrings = set(all_substrings)#.union(set([("",)]))
    # don't look at all substrings for adjacent
    all_substrings = list(filter(lambda x: len(x) < 3, all_substrings))
    L = len(substrings)
    for i in range(L):
        u = substrings[i]
        for j in range(i+1, L):
            v = substrings[j]
            # l and r in sigma*
            # but wouldnt this make all u,v substitutable? should be sigma+?
            # no, need it for ends, i think for ends I may have to add ending character, #
            if u != v:
                for l in all_substrings:
                    for r in all_substrings:
                        if l[0] != "" or r[0] != "":
                            sequence1 = make_new_sequence([l, u, r])
                            if sequence_in_T(sequence1, T):
                                sequence2 = make_new_sequence([l, v, r])
                                if sequence_in_T(sequence2, T):
                                    #print(l, u, r)
                                    #print(l, v, r)
                                    #print()
                                    SG.add_edge(u, v)

    # remove all nodes with degree of 0
    for node in SG.nodes():
        print(node, SG.degree(node))
        if SG.degree(node) == 0:
            SG.remove_node(node)
    return SG




# Alphabet is a set of tokens
# Non-Terminals is a set of tokens of form "NT#"
# Production rules is a dictionary where the key is a non-terminal
# and the value is a list of possible rules the non-terminal can become
# S is a set of all the non-terminal start symbols
# T is a list. Each element in the list is a sequence of tokens where
# each token is either a member of the alphabet or non-terminal set

# keys of productions are strings representing NONTERMINALS only
def LearnCFG(T):
    sigma = get_alphabet(T)
    #G = (alphabet, non-terminals, production rules, start symbol (S, non-terminal))
    S = "NT0"
    G = [sigma, set([S]), {}, set([S])]
    #while T not in G:
    for i in range(11):
        print("Stage:", i+1, "/20")

        Tp = T
        print()
        print("Productions")
        print(G[2])
        for w in Tp:
            print(w)
        print()
        G = ExtendGrammar(G, T)
        T = GreedyReduceData(G, T)

        if set(list(map(tuple, T))) == set(list(map(tuple, Tp))):
            break
    print("Learned Grammar")
    print(G)
    return G


# Something is wrong. The only productions I am creating have a rule with just one token
# That should be allowed to happen, but how do I get larger rules?
# problem is that the mutual information of all sequences are 0, len +1
def ExtendGrammar(G, T):
    # get substrings that are likely constituents
    SG = construct_substitution_graph(G, T)
    Gp = np.copy(G)
    # component means connected components
    sigma, NTs, P, S = Gp
    for component in nx.connected_components(SG):
        # check all nodes in C
        # if there is a single non-terminal as one of the nodes in C return it
        # otherwise return None
        A = single_NT(component)
        print("component", component)
        #print("A ", A)
        if A != False:
            for node in component:
                if node != A:
                    # add production rule. A -> node
                    # a node is a sequence and must be converted in funtion
                    assert type(A) == tuple
                    assert len(A) == 1
                    P = add_production_rule(P, A[0], node)
        else:
            # add new non-terminal, combine N and S sets
            NT = generate_new_NT(Gp)
            for node in component:
                P = add_production_rule(P, NT, node)
            NTs = NTs.union(set([NT]))
            Gp = (sigma, NTs, P, S)


    l, f = 4, 2
    freq = get_freq(T)
    #print(freq)
    for example in T:
        sequence = make_new_sequence(example)
        if len(sequence) == 1 and sequence[0][:2]=="NT":
            if sequence[0] not in S:
                S   = S.union(set([sequence[0]]))
        elif len(sequence) < l and freq[sequence] > f:
            # generate new start non terminal
            # combine N and S sets

            Si = generate_new_NT(Gp)
            P = add_production_rule(P, Si, sequence)
            NTs = NTs.union(set([Si]))            
            S   = S.union(set([Si]))
            print("Si,", Si, sequence)            
            Gp = (sigma, NTs, P, S)
            break

    return (sigma, NTs, P, S)



def add_production_rule(P, nonterminal, sequence):
    #print("sequence,", sequence)
    assert type(sequence) == tuple
    assert type(P) == dict
    #assert type(list(P.values())[0]) == list
    #assert type(list(P.values())[0][0]) == tuple
    #assert len(list(P.values())[0][0]) == 1

    # turn sequence into rule
    new_rule = []
    for symbol in sequence:
        assert type(symbol) == str
        new_rule.append((symbol,))

    duplicate = False
    if nonterminal in P.keys():
        for rule in P[nonterminal]:
            if rule == new_rule:
                duplicate = True
                break
    if nonterminal in P.keys():
        if not duplicate:
            P[nonterminal].append(new_rule)
    else:
        P[nonterminal] = []
        P[nonterminal].append(new_rule)
    return P


def GreedyReduceData(G, T):
    Tp = []
    for w in T:
        Tp.append(alpha(w, G))
    return Tp


def alpha(w, G):
    assert type(w) == list
    if len(w) > 0:
        assert type(w[0]) == tuple
        assert len(w[0]) ==1
    sigma, nts, P, s = G
    possible_reductions = reduce(w, P)
    if len(possible_reductions) > 0:
        lengths = list(map(len, possible_reductions))
        index = np.argmin(lengths)
        # only substitute if length of reduction is smaller than
        # original length
        if len(possible_reductions[index]) <= len(w):
            w = possible_reductions[index]
    return w


#@memoize
def reduce(tokens, P):
    reductions = []
    seq_len = len(tokens)
    for nt in P.keys():
        for rule in P[nt]:
            rule_len = len(rule)
            for i in range(seq_len-rule_len+1):
                if tokens[i:i+rule_len] == rule:
                    wr = tokens[:i] + [(nt,)] + tokens[i+rule_len:]
                    reductions.append(wr)
                    reductions.extend(reduce(wr, P))
    return reductions





def handmade_test(n=0):
    num_examples = 30
    print("Generate Training and Testing Sets")
    T_train = None
    while T_train is None:
        try:
            Gt = generate_handmade_grammar(n)
            T_train = generate_positive_examples(Gt, num_examples)
        except:
            pass

    G = LearnCFG(T_train)
    print()
    print("Grammar Learned")
    print( G)


def english_test():
    T_train = [" This ", " That ", "I am a man", "I am a woman"]
    T_train = [list(map(lambda x: (x,), example)) for example in T_train]
    G = LearnCFG(T_train)
    print("Grammar Learned")
    print( G )


def python_test():
    T_train = generate_python()
    G = LearnCFG(T_train)
    print("Grammar Learned")
    print( G )


def haskell_test():
    T_train = generate_haskell()
    G = LearnCFG(T_train)
    print("Grammar Learned")
    print( G )


# Quantitative Test that could not be performed
def random_test():
    generated_grammars = []
    num_examples = 150
    print(.2*num_examples)
    Ne_te, Ne_tr = 0, 0
    print("Generate Training and Testing Sets")
    for i in range(15):
        try:
            Gt = generate_random_grammar()
            T = generate_positive_examples(Gt, num_examples)

            T_train = T[:int(.2*num_examples) ]
            T_test  = T[ int(.2*num_examples):]
            
            #print("Train", T_train)
            generated_grammars.append((Gt, T_train, T_test))
            Ne_te += len(T_test)
            Ne_tr += len(T_train)
        except:
            pass
    Ng = len(generated_grammars)
    s = 0
    for g,_,_ in generated_grammars:
        s += len(g[0]) * len(g[1])
    print("Omphalos Competition Size: ", s)
    print("Size of Training Set: ", Ng*int(.2*num_examples))
    print("Size of Test Set:     ", Ng*int(.8*num_examples))

    train_errors = []
    test_errors  = []
    Gs = []
    print("Run Training Algorithm")
    for i in range(len(generated_grammars)):
        print("Grammars Trained: ", i+1, "/", len(generated_grammars))
        Gt, T_train, T_test = generated_grammars[i]
        try:
            Gl = LearnCFG(T_train)
            print()
            print("Grammar Learned")
            print( Gl)
            Gs.append(Gl)

            e1 = get_correct(Gl, T_train)
            e2 = get_correct(Gl, T_test)
            print("Train Error: ", e1)
            print("Test Error: ", e2)

            train_errors.append(e1)
            test_errors.append(e2)
        except Exception as e:
            print(e)
            print("Error in Learning")
    print()
    print()
    print("Errors")
    print(train_errors)
    print(test_errors)
    train_ga, train_ea = get_acc(train_errors, Ne_te)
    test_ga,  test_ea  = get_acc(test_errors, Ne_tr)
    print("Training Grammar Accuracy: ", train_ga)
    print("Training Example Accuracy: ", train_ea)
    print()
    print("Testing Grammar Accuracy:  ", test_ga)
    print("Testing Example Accuracy:  ", test_ea)



if __name__ == "__main__":
    handmade_test()
    english_test()
    #python_test()
    #haskell_test()
    #random_test()
