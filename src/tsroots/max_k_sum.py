import numpy as np
import heapq

class Combination:
    """
    Combination class to extract columm indices for the current combination,
    their correspondng values, and sum of the values.
    """
    def __init__(self, indices, values, summation):
        self.indices = indices
        self.values = values
        self.summation = summation

    def __lt__(self, other):
        return self.summation < other.summation

    def __repr__(self):
        # return f'Sum: {self.summation:.2f}, Indices: {self.indices}, Values: {self.values}'
        return f'{self.indices}'

    def __iter__(self):
        # return f'Sum: {self.summation:.2f}, Indices: {self.indices}, Values: {self.values}'
        return iter((f'{self.indices}'))

    def __int__(self):
        return self.indices

    def __getitem__(self, item):
        return self.indices[item]


def find_max_k_sum_without_dp(matrix, k):
    """
    Implementing maxk_sum via heap data structure to avoid the curse of dimensionality
    that arise for the roots combinations of several roots in higher dimensions
    """
    combinations = []
    heapq.heapify(combinations)

    # print("Step 1: Initialize heap with combinations from the first row.")
    for col, num in enumerate(matrix[0]):
        comb = Combination([int(col)], [num], num)
        # print(comb)
        heapq.heappush(combinations, comb)
        if len(combinations) > k:
            heapq.heappop(combinations)
    # print(combinations)
    # print_heap(combinations, "Initial heap")

    # For subsequent rows, create new combinations and push them into the heap
    for row in range(1, len(matrix)):
        next_combinations = []
        # print(f"\nStep 2: Process row {row}")

        for comb in combinations:
            for col, num in enumerate(matrix[row]):
                new_indices = comb.indices + [int(col)]
                new_values = comb.values + [num]
                new_summation = comb.summation + num
                new_comb = Combination(new_indices, new_values, new_summation)

                heapq.heappush(next_combinations, new_comb)
                if len(next_combinations) > k:
                    heapq.heappop(next_combinations)

        combinations = next_combinations
        # print_heap(combinations, f"Heap after row {row}")

    # Collect the final top k combinations from the heap, reversed for descending order
    result = sorted(combinations, reverse=True)

    return result

# Utility function to print the top element of the heap
def print_heap(heap, message=""):
    print(f"{message} - Entire heap:")
    for item in heap:
        print(item)

# Generate a typical samples for maxk_sum using logarithimic sorting.
def samples(n,d):
    A = np.zeros((n,d))
    for i in range(d):
        r = np.random.uniform(size=n)
        #print(r)
        A[:,i] = np.log(sorted(r, reverse=True) / max(r))
    return A

if __name__ == "__main__":
    n = 5
    d = 4
    matrix = samples(n, d)
    print(matrix)

    # print top combinations of the sample matrix generated above:
    top_combinations = find_max_k_sum_without_dp(matrix, 2)
    print(top_combinations)