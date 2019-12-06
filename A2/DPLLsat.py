#!/usr/bin/python3
# CMPT310 A2
#####################################################
#####################################################
# Please enter the number of hours you spent on this
# assignment here
"""
num_hours_i_spent_on_this_assignment = 35
"""
#
#####################################################
#####################################################

#####################################################
#####################################################
# Give one short piece of feedback about the course so far. What
# have you found most interesting? Is there a topic that you had trouble
# understanding? Are there any changes that could improve the value of the
# course to you? (We will anonymize these before reading them.)
"""
<Your feedback goes here>
Appreciate the prof's effort to improve. Looking forward to learn more about AI.

"""
#####################################################
#####################################################
import sys, getopt
import copy
import random
import time
import numpy as np
sys.setrecursionlimit(10000)

class SatInstance:
    def __init__(self):
        pass

    def from_file(self, inputfile):
        self.clauses = list()
        self.VARS = set()
        self.p = 0
        self.cnf = 0
        with open(inputfile, "r") as input_file:
            self.clauses.append(list())
            maxvar = 0
            for line in input_file:
                tokens = line.split()
                if len(tokens) != 0 and tokens[0] not in ("p", "c"):
                    for tok in tokens:
                        lit = int(tok)
                        maxvar = max(maxvar, abs(lit))
                        if lit == 0:
                            self.clauses.append(list())
                        else:
                            self.clauses[-1].append(lit)
                if tokens[0] == "p":
                    self.p = int(tokens[2])
                    self.cnf = int(tokens[3])
            assert len(self.clauses[-1]) == 0
            self.clauses.pop()
            if (maxvar > self.p):
                print("Non-standard CNF encoding!")
                sys.exit(5)
        # Variables are numbered from 1 to p
        for i in range(1, self.p + 1):
            self.VARS.add(i)

    def __str__(self):
        s = ""
        for clause in self.clauses:
            s += str(clause)
            s += "\n"
        return s


def main(argv):
    inputfile = ''
    verbosity = False
    inputflag = False
    try:
        opts, args = getopt.getopt(argv, "hi:v", ["ifile="])
    except getopt.GetoptError:
        print('DPLLsat.py -i <inputCNFfile> [-v] ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('DPLLsat.py -i <inputCNFfile> [-v]')
            sys.exit()
        ##-v sets the verbosity of informational output
        ## (set to true for output veriable assignments, defaults to false)
        elif opt == '-v':
            verbosity = True
        elif opt in ("-i", "--ifile"):
            inputfile = arg
            inputflag = True
    if inputflag:
        instance = SatInstance()
        instance.from_file(inputfile)
        #start_time = time.time()
        solve_dpll(instance, verbosity)
        #print("--- %s seconds ---" % (time.time() - start_time))

    else:
        print("You must have an input file!")
        print('DPLLsat.py -i <inputCNFfile> [-v]')

# Selecting the variable that appears most frequently
def pick_a_variable(clauses):
    varFreq = dict()
    max, variable = 0, 0
    for c in clauses:
        for element in c:
            if element in varFreq:
                varFreq[element] += 1
                if max < varFreq[element]:
                    max = varFreq[element]
                    variable = element
            else:
                varFreq[element] = 1
    return variable

# Count number of occurrances of each variable
def count(clauses): 
    counter = {}
    for c in clauses:
        for x in c:
            if x in counter:
                counter[x] += 1
            else:
                counter[x] = 1
    return counter

# Remove clauses with symbol
def removeClause(clauses, symbol):
    updatedClauses = []
    for c in clauses:
        if symbol in c: continue 
        if -symbol in c:
            clause = []
            for x in c:
                if x != -symbol:
                    clause.append(x)
            if len(clause) == 0: 
                return -1 
            updatedClauses.append(clause)
        else:
            updatedClauses.append(c)
    return updatedClauses

# Unit Propagation
def propagate_units(clauses):
    vars = []
    unit_clauses = [x for x in clauses if len(x) == 1] 
    while len(unit_clauses) > 0: 
        unit = unit_clauses[0]
        clauses = removeClause(clauses, unit[0]) 
        vars += [unit[0]]
        if clauses == -1:
            return -1, []
        if not clauses:
            return clauses, vars
        unit_clauses = [x for x in clauses if len(x) == 1] 
    return clauses, vars

# Pure Literal Elimination
def pure_elim(clauses):  
    vars = []
    pure = []
    counter = count(clauses)
    for variable, occurances in counter.items():
        if -variable not in counter: 
            pure.append(variable) 
    for literal in pure:
        clauses = removeClause(clauses, literal) 
    vars += pure
    return clauses, vars

# DPLL SAT Solver
def DPLL(model, clauses):
    clauses, pure = pure_elim(clauses) #Pure Literal Elimination
    clauses, unit = propagate_units(clauses) #Unit Propagation

    model += (unit + pure)
    if clauses == -1:
        return []
    if not clauses:
        return model
    
    #Variable Selection
    var = pick_a_variable(clauses)
    
    #Recursive Backtracking
    ret = DPLL(model + [var], removeClause(clauses, var))
    if not ret:
        ret = DPLL(model + [-var], removeClause(clauses, -var))
    return ret

# Output list of true literals
def format_output(symbols):
    trueLiterals = []
    
    for literal in symbols:
        if literal > 0:
            trueLiterals.append(literal)
    trueLiterals.sort()
    print(str(trueLiterals))

# Finds a satisfying assignment to a SAT instance,
# using the DPLL algorithm.
# Input: a SAT instance and verbosity flag
# Output: print "UNSAT" or
#    "SAT"
#    list of true literals (if verbosity == True)
#
#  You will need to define your own
#  DPLLsat(), DPLL(), pure-elim(), propagate-units(), and
#  any other auxiliary functions

def solve_dpll(instance, verbosity):
    # print(instance)
    # instance.VARS goes 1 to N in a dict
    # print(instance.VARS)
    # print(verbosity)
    ###########################################
    # Start your code
    clauses = instance.clauses
    model = []
    result = DPLL(model, clauses)
    
    if result:
        print("SAT")
        if verbosity == True:
            format_output(result)
    else:
        print("UNSAT")
    ###########################################


if __name__ == "__main__":
    main(sys.argv[1:])

#    REFERENCES:
# https://cs.brown.edu/courses/cs195y/2017/labs/lab-4.html
