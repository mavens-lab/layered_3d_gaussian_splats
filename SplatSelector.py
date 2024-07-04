import typing
import numpy as np
import time
import math

#SplatSelector takes a list of objects and layers and their utility and size as input, and the available capacity.
#It outputs the selected layers that maximize utility within the time slot.
#Suppose we have N objects and M layers.
#Example execution code is at the bottom of this script.

#inputs:
#  utility: NxM size matrix, with integer utility values.
#  size: NxM matrix, with the size of each layer and object
#  budget: the available capacity

#outputs:
#  selected_layers: NxM size matrix, with "1" meaning the layer should be downloaded, and "0" otherwise.

class SplatSelector:

    def __init__(self,utility,size,budget):
        num_objects = len(utility)
        num_layers = len(utility[0])

        #do a bit of data manipulation for the multiple choice knapsack code, which has a different data input format
        self.items = list(range(num_objects*num_layers))
        self.values = self.__flatten_concatenation(utility)       
        #self.weights = self.__flatten_concatenation([list(range(i,i+num_layers*i,i)) for i in size] #this is for the old version where size was a Mx1 vector
        self.weights = self.__flatten_concatenation(size)
        self.groups = self.__flatten_concatenation([[i]*num_layers for i in range(num_objects)])
        self.budget = budget

    # private helper function to flatten a list of lists into a single list
    def __flatten_concatenation(self,matrix):
        flat_list = []
        for row in matrix:
            flat_list += row
        return flat_list

    # private function to solve multiple choice knapsack problem
    # source: https://gist.github.com/USM-F/1287f512de4ffb2fb852e98be1ac271d
    def __solve_multiple_choice_knapsack(self,
        items: typing.List[object],
        capacity: int,
        weights: typing.List[int],
        values: typing.List[int],
        groups: typing.List[int],
        noLog: bool = True,
        forceLongRun: bool = False
    )-> typing.Tuple[int, typing.List[object]]:
        """
        Solves knapsack where you need to knapsack a bunch of things, but must pick at most one thing from each group of things
        #Example
        items = ['a', 'b', 'c']
        values = [60, 100, 120]
        weights = [10, 20, 30]
        groups = [0, 1, 1]
        capacity = 50
        maxValue, itemList = solve_multiple_choice_knapsack(items, capacity, weights, values, groups)

        Extensively optimized by Travis Drake / EklipZgit by an order of magnitude, original implementation cloned from: https://gist.github.com/USM-F/1287f512de4ffb2fb852e98be1ac271d

        @param items: list of the items to be maximized in the knapsack. Can be a list of literally anything, just used to return the chosen items back as output.
        @param capacity: the capacity of weights that can be taken.
        @param weights: list of the items weights, in same order as items
        @param values: list of the items values, in same order as items
        @param groups: list of the items group id number, in same order as items. MUST start with 0, and cannot skip group numbers.
        @return: returns a tuple of the maximum value that was found to fit in the knapsack, along with the list of optimal items that reached that max value.
        """

        start = time.time()
        timeStart = time.perf_counter()
        groupStartEnds: typing.List[typing.Tuple[int, int]] = []
        if groups[0] != 0:
            raise AssertionError('Groups must start with 0 and increment by one for each new group. Items should be ordered by group.')

        lastGroup = -1
        lastGroupIndex = 0
        maxGroupSize = 0
        curGroupSize = 0
        for i, group in enumerate(groups):
            if group > lastGroup:
                if curGroupSize > maxGroupSize:
                    maxGroupSize = curGroupSize
                if lastGroup > -1:
                    groupStartEnds.append((lastGroupIndex, i))
                    curGroupSize = 0
                if group > lastGroup + 1:
                    raise AssertionError('Groups must have no gaps. if you have group 0, and 2, group 1 must be included between them.')
                lastGroupIndex = i
                lastGroup = group

            curGroupSize += 1

        groupStartEnds.append((lastGroupIndex, len(groups)))
        if curGroupSize > maxGroupSize:
            maxGroupSize = curGroupSize

        # if BYPASS_TIMEOUTS_FOR_DEBUGGING:
        for value in values:
            if not isinstance(value, int):
                raise AssertionError('values are all required to be ints or this algo will not function')

        n = len(values)
        K = [[0 for x in range(capacity + 1)] for x in range(n + 1)]
        """knapsack max values"""

        maxGrSq = math.sqrt(maxGroupSize)
        estTime = n * capacity * math.sqrt(maxGroupSize) * 0.00000022
        """rough approximation of the time it will take on MY machine, I set an arbitrary warning threshold"""
        if maxGroupSize == n:
            # this is a special case that behaves like 0-1 knapsack and doesn't multiply by max group size at all, due to the -1 check in the loop below.
            estTime = n * capacity * 0.00000022

        if estTime > 0.010 and not forceLongRun:
            raise AssertionError(f"The inputs (n {n} * capacity {capacity} * math.sqrt(maxGroupSize {maxGroupSize}) {maxGrSq}) are going to result in a substantial runtime, maybe try a different algorithm")
        if not noLog:
            logging.info(f'estimated knapsack time: {estTime:.3f} (n {n} * capacity {capacity} * math.sqrt(maxGroupSize {maxGroupSize}) {maxGrSq})')

        for curCapacity in range(capacity + 1):
            for i in range(n + 1):
                if i == 0 or curCapacity == 0:
                    K[i][curCapacity] = 0
                elif weights[i - 1] <= curCapacity:
                    sub_max = 0
                    prev_group = groups[i - 1] - 1
                    subKRow = curCapacity - weights[i - 1]
                    if prev_group > -1:
                        prevGroupStart, prevGroupEnd = groupStartEnds[prev_group]
                        for j in range(prevGroupStart + 1, prevGroupEnd + 1):
                            if groups[j - 1] == prev_group and K[j][subKRow] > sub_max:
                                sub_max = K[j][subKRow]
                    K[i][curCapacity] = max(sub_max + values[i - 1], K[i - 1][curCapacity])
                else:
                    K[i][curCapacity] = K[i - 1][curCapacity]

        res = K[n][capacity]
        timeTaken = time.perf_counter() - timeStart
        if not noLog:
            logging.info(f"Value Found {res} in {timeTaken:.3f}")
        includedItems = []
        includedGroups = []
        w = capacity
        lastTakenGroup = -1
        for i in range(n, 0, -1):
            if res <= 0:
                break
            if i == 0:
                raise AssertionError(f"i == 0 in knapsack items determiner?? res {res} i {i} w {w}")
            if w < 0:
                raise AssertionError(f"w < 0 in knapsack items determiner?? res {res} i {i} w {w}")
            # either the result comes from the
            # top (K[i-1][w]) or from (val[i-1]
            # + K[i-1] [w-wt[i-1]]) as in Knapsack
            # table. If it comes from the latter
            # one/ it means the item is included.
            # THIS IS WHY VALUE MUST BE INTS
            if res == K[i - 1][w]:
                continue

            group = groups[i - 1]
            if group == lastTakenGroup:
                continue

            includedGroups.append(group)
            lastTakenGroup = group
            # This item is included.
            if not noLog:
                logging.info(
                    f"item at index {i - 1} with value {values[i - 1]} and weight {weights[i - 1]} was included... adding it to output. (Res {res})")
            includedItems.append(items[i - 1])

            # Since this weight is included
            # its value is deducted
            res = res - values[i - 1]
            w = w - weights[i - 1]

        uniqueGroupsIncluded = set(includedGroups)
        if len(uniqueGroupsIncluded) != len(includedGroups):
            raise AssertionError("Yo, the multiple choice knapsacker failed to be distinct by groups")

        if not noLog:
            logging.info(
                f"multiple choice knapsack completed on {n} items for capacity {capacity} finding value {K[n][capacity]} in Duration {time.perf_counter() - timeStart:.3f}")

        print("Time elapsed:", round((time.time() - start)*1000,2), "ms")
        
        #JC: convert includedItems to the format that we want
        selected_layers = np.zeros((int(n/maxGroupSize), maxGroupSize), dtype=int)
        for i in includedItems:
            selected_layers[math.floor(i/maxGroupSize),0:i%maxGroupSize+1] = 1
        
        
        #return K[n][capacity], includedItems
        return selected_layers.tolist()

    #public function to solve multiple choice knapsack
    def solve_multiple_choice_knapsack(self):
        return self.__solve_multiple_choice_knapsack(self.items, self.budget, self.weights, self.values, self.groups)



#example code to test is below

#generate random increasing utility functions
util = np.random.rand(5,4)
util[:,1] = util[:,1] + util[:,0]
util[:,2] = util[:,2] + util[:,1]
util[:,3] = util[:,3] + util[:,2] 
util = [[round(i*100)+1 for i in nested] for nested in util] #round up utility to integers for knapsack problem

print("utility of objects:", util)

#generate random sizes
#size = [1, 2, 3, 4, 5]
size = np.random.rand(5,4)
size = [[round(i*10+1) for i in nested] for nested in size]
print("Size of each object each layer:", size)

# predicted bandwidth
budget = 10
print("Budget:", budget)

#solve the knapsack
ss = SplatSelector(util,size,budget)
print()
print(ss.solve_multiple_choice_knapsack())
