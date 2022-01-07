# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 06:13:27 2021

@author: Tashfeen Mustafa Choudhury

Program: FP-Growth Algorithm for finding frequent itemsets
"""

import sys
import time
import itertools
import numpy as np


def load_dataset(dataset_name):
    path = 'E:\IUB-Courses\Autumn 2021\Data Mining & Warehouse\Assignment 3 - FrequentItemset\FP_Growth\data'
    input_file_path = path + '\\' + dataset_name
    data = None
    try:
        data = np.loadtxt(input_file_path, dtype='int64')
    except:
        # if baskets are uneven
        # read data like this
        data = []
        file = open(input_file_path)

        for line in file:
            line = line.strip().split(' ')
            temp = []
            for l in line:
                temp.append(int(l))
            data.append(temp)

    return data


def output_pattern(pattern, item, dataset_name):
    file_name = 'result_' + dataset_name[:-4] + '.txt'

    mode = 'a'

    file = open(file_name, mode)

    file.write('item-' + str(item) + '\n' + str(pattern) + '\n')

    file.close()


def output_txt(l_k, dataset_name, k):
    file_name = 'result_' + dataset_name[:-4] + '.txt'
    mode = ''
    if k == 1:
        mode = 'w'
    else:
        mode = 'a'

    file = open(file_name, mode)

    file.write('L-' + str(k) + '\n' + str(l_k) + '\n')

    file.close()


def print_summary(data):
    average_transaction_length = 0

    for d in data:
        average_transaction_length += len(d)

    average_transaction_length /= len(data)

    print('Total Number of Transactions: ', len(data))
    print('Average Transaction Length: ', average_transaction_length)


def find_frequent_l_items(data, min_sup):
    itemset = None

    # create first candidate list (c_1)
    c_1 = []

    # loops through data and updates l_1 with itemset
    for d in data:
        for j in range(len(d)):
            # print('j: ', j)
            itemset = d[j]

            # assuming itemset is new item
            new_item = True

            c_1_length = len(c_1)

            if c_1_length > 0:

                # check if itemset exists
                for i in range(c_1_length):
                    # print('itemset: ', itemset)
                    # print('c_1[i][0]: ', c_1[i][0])
                    # print('i: ', i)
                    if itemset == c_1[i][0]:
                        # if it does increment itemset count
                        # print('here')
                        c_1[i][1] += 1
                        new_item = False

            # if itemset is new item append to c_1
            if new_item:
                c_1.append([itemset, 1])
            # print(c_1)

    # create l_1 frequent itemsets
    l_1 = []

    for c in c_1:
        # only add the itemsets that have count >= min_sup
        if c[1] >= min_sup:
            l_1.append(c)

    # sort the list
    l_1 = sorted(l_1, key=lambda x: [-x[1], x[0]])

    return l_1


class FPTreeNode:
    # Function to initialize the node object
    def __init__(self, item, parent=None):
        self.item = item  # Assign item
        self.count = 1  # Initialize count with 1
        self.parent = parent  # Every FPTreeNode will always have one parent node except for the root node
        self.children = []  # child nodes of an FPTreeNode (could be N number of children)
        self.next = None  # Initialize next as None


class LinkedList:
    # Initialize Linked List object
    def __init__(self):
        self.head = None

    def insert_fptree_node_to_the_end(self, fptree_node):
        # Check if the head is empty
        # if empty, add the first node to the head and update the tail
        # else create a new node and add it to the tail and then update
        # the tail 
        if self.head is None:
            self.head = fptree_node
        else:
            prev = self.head
            curr = self.head
            while curr:
                prev = curr
                curr = curr.next
            prev.next = fptree_node

    def get_count(self):
        count = 0
        curr = self.head

        while curr:
            count += curr.count
            curr = curr.next

        return count

    def show_list(self):
        # Loop through the list and print each data
        temp = self.head
        while temp:
            print((temp.item, temp.count), end='->')
            temp = temp.next
            if not temp:
                print(None)

    def pop(self):
        # delete data from the end of the linked list
        curr = self.head
        prev = self.head
        while curr:
            prev = curr
            if prev.next.next is None:
                temp = prev.next
                temp = None
                prev.next = temp
            curr = curr.next


# noinspection PyMethodMayBeStatic
class FPTree:
    def __init__(self, fptree_root):
        self.root = fptree_root

    def add_prefix_to_fptree(self, prefix, fptree_node, linked_list):
        # takes prefix and adds the prefix to the tree

        last_fptree_node = None

        # check if fp_tree node has children
        if len(fptree_node.children) > 0:
            # if it does, check if children have a node with the same prefix
            for child in fptree_node.children:
                if child.item == prefix:
                    # if it does then increase count of that item
                    child.count += 1

                    # send address of this node
                    last_fptree_node = child
                    return last_fptree_node

            # if the children of the fptree node doesn't have a node with the same prefix
            # create a FPTreeNode for the prefix
            new_fptree_node = FPTreeNode(prefix, fptree_node)

            # add the new fptree node to the fptree node's children
            fptree_node.children.append(new_fptree_node)

            # add node to corresponding linked list
            linked_list.insert_fptree_node_to_the_end(new_fptree_node)

            # send address of this node
            last_fptree_node = new_fptree_node
        elif len(fptree_node.children) == 0:
            # if root of fp tree has no children, add the item as an fptree node to the root
            # create a FPTreeNode for the prefix
            new_fptree_node = FPTreeNode(prefix, fptree_node)

            # add the new fptree node to the fptree node's children
            fptree_node.children.append(new_fptree_node)

            # add node to corresponding linked list
            linked_list.insert_fptree_node_to_the_end(new_fptree_node)

            # send address of this node
            last_fptree_node = new_fptree_node

        return last_fptree_node

    def visualize_fptree(self):
        print('Printing fp-tree: ')
        # create frontier for fptree to track all children of each node
        # traversing tree in bfs fashion
        frontier = []

        # append the root to frontier
        frontier.append(self.root)

        # while frontier is not empty, dequeue node
        while frontier:
            fp_node = frontier.pop(0)

            # print the item, count, and node
            print((fp_node.item, fp_node.count, '#' if fp_node.parent is None else fp_node.parent.item))

            # add every child node to frontier
            for child in fp_node.children:
                frontier.append(child)

        return None

    def get_length(self):
        length = 1

        cursor = self.root

        while cursor:
            if len(cursor.children) > 0:
                length += 1
                cursor = cursor.children[0]
            else:
                break

        return length


def join_conditional_fptrees(all_cond_fptrees):
    longest_fptree_index = 0
    longest_fptree_length = 0

    # get index of longest fp tree
    for index, fptree in enumerate(all_cond_fptrees):
        if fptree.get_length() > longest_fptree_length:
            longest_fptree_length = fptree.get_length()
            longest_fptree_index = index

    # print(longest_fptree_index)

    # take longest fp tree as final fp tree and join other fp trees to this one
    final_fptree = all_cond_fptrees[longest_fptree_index]
    final_fptree_cursor = final_fptree.root
    final_fptree.visualize_fptree()

    for i in range(len(all_cond_fptrees)):
        fptree = None
        fptree_cursor = None

        if i != longest_fptree_index:
            fptree = all_cond_fptrees[i]
            fptree_cursor = fptree.root
            fptree.visualize_fptree()

        is_item_equal = False
        while fptree_cursor:
            # print(fptree_cursor.item)

            if final_fptree_cursor.item == fptree_cursor.item:
                is_item_equal = True
                final_fptree_cursor.count = final_fptree_cursor.count + fptree_cursor.count

            if fptree_cursor.children:
                if is_item_equal:
                    fptree_cursor = fptree_cursor.children[0]
                    final_fptree_cursor = final_fptree_cursor.children[0]
                else:
                    final_fptree_cursor = final_fptree_cursor.children[0]
            else:
                if is_item_equal:
                    fptree_cursor = fptree_cursor.children
                else:
                    final_fptree_cursor = final_fptree_cursor.children[0]

            is_item_equal = False

        final_fptree_cursor = final_fptree.root

    return final_fptree


def prune_conditional_fptree(joined_conditional_fptree, min_sup):
    # loop through the fp tree and go to the end of the fp tree
    fptree_cursor = joined_conditional_fptree.root
    last_fp_node = None

    while fptree_cursor:
        # print(fptree_cursor.item)
        if fptree_cursor.children:
            fptree_cursor = fptree_cursor.children[0]
        else:
            last_fp_node = fptree_cursor
            fptree_cursor = fptree_cursor.children

    # print(last_fp_node.item)

    # travel up the tree and remove the items with count < min_sup
    # stop travelling if the count > min_sup
    fptree_cursor = last_fp_node

    while fptree_cursor:
        # print(fptree_cursor.item)
        next_parent = None
        if fptree_cursor.count < min_sup:
            next_parent = fptree_cursor.parent
            if next_parent:
                next_parent.children = []

        fptree_cursor = next_parent

    joined_conditional_fptree.visualize_fptree()

    return joined_conditional_fptree


def generate_subsets(item, fp_tree):
    all_subsets = []

    # create a set with each item of fp_tree then the whole fp tree
    fptree_cursor = fp_tree.root
    last_fp_node = None
    temp_set = list()
    fptree_length = fp_tree.get_length()

    while fptree_cursor:
        # print(fptree_cursor.item)
        if fptree_cursor.item:
            temp_set.append(fptree_cursor.item)

        if fptree_cursor.children:
            fptree_cursor = fptree_cursor.children[0]
        else:
            last_fp_node = fptree_cursor
            fptree_cursor = fptree_cursor.children

    for i in range(1, fptree_length - 1):
        subset = list(itertools.combinations(temp_set, i))

        new_subset = []

        for sub in subset:
            s = list(sub)
            s.append(item)
            print(s)
            new_subset.append([tuple(s), last_fp_node.parent.count])

        all_subsets.append(new_subset)

    return all_subsets


def create_fp_tree_from_conditional_pattern_base(conditional_pattern_base):
    # create root for the conditional fp tree
    temp_fptree_node = None

    # first fp node (bottom first)
    previous_fptree_node = None

    for i in range(len(conditional_pattern_base)):
        item = conditional_pattern_base[i][0]
        count = conditional_pattern_base[i][1]

        # create fptree node
        temp_fptree_node = FPTreeNode(item)
        temp_fptree_node.count = count

        if i == 0:
            # add fptree node to previous fptree node
            previous_fptree_node = temp_fptree_node
        else:
            # add previous fp tree node as child of temp fp tree node
            temp_fptree_node.children.append(previous_fptree_node)
            previous_fptree_node.parent = temp_fptree_node
            previous_fptree_node = temp_fptree_node

    # create fp tree
    cond_fp_tree = FPTree(temp_fptree_node)
    return cond_fp_tree


def find_conditional_pattern_base(fptree_node):
    head = fptree_node

    conditional_pattern_base = []

    while head:
        conditional_pattern_base.append((head.item, head.count))
        head = head.parent

    return conditional_pattern_base


def fp_growth(data, min_sup, dataset_name):
    ###### Step 1
    l_1 = find_frequent_l_items(data, min_sup)

    print('l_1: ', l_1)
    print('\n')

    output_txt(l_1, dataset_name, k=1)

    ###### Step 2
    # to make sorting each transaction (t) in data in order of
    # l_1 easier we will take a list of items in l_1 without the 
    # count for each item. This is because we do not need the count
    # for each item when scanning the data and sorting each transaction (t)
    l_1_without_count = []

    for l in l_1:
        l_1_without_count.append(l[0])

    # scan data again and for every transaction (t) in data
    # sort t in order of l_1

    sorted_transaction_db = []

    for i in range(len(data)):
        # taking each transaction (t)
        t = data[i]
        new_t = []

        # checking if item in transaction (t) is in l_1_without_count
        # if not, no point in having that item in the new sorted transaction
        # if a whole transaction (t) doesnot have any of the item in l_1 itemset
        # then skip the transaction
        for item in t:
            if item in l_1_without_count:
                new_t.append(item)

        # sort the new_t based on l_1_without_count
        new_t.sort(key=lambda x: l_1_without_count.index(x))
        new_t_sorted = new_t

        # add the new_t to sorted_transaction_db
        sorted_transaction_db.append(new_t_sorted)

    for t in sorted_transaction_db:
        print(t)
    print(len(sorted_transaction_db))
    print('l_1: ', l_1)

    ###### Step 3
    # now construct an fp-tree based on the sorted_transaction_db
    # and also maintain the auxillary data structure side by side
    # auxillary data structure: [{id: 1, count: 5000, linked_list: <class LinkedList>}]

    # create root node of FPTree and initialize it
    fptree_root = FPTreeNode(None)

    # add root node to fp tree
    fp_tree = FPTree(fptree_root)

    # Create auxillary data structure
    aux_ds = []

    # for each l in l_1 create a temp_ds and append to aux_ds
    for l in l_1:
        temp_ds = {"id": l[0], "count": 0, "linked_list": LinkedList()}
        aux_ds.append(temp_ds)

    # now traverse the sorted_transaction_db
    for t in sorted_transaction_db:
        # search from the fptree root in every iteration
        fptree_node = fp_tree.root
        fptree_node_linkedlist = None

        for pattern_item in t:
            # take the pattern
            prefix = pattern_item

            # check if the prefix matches with some row in aux_ds
            for row in aux_ds:
                if row["id"] == prefix:
                    # if so, take the linked list of that row
                    fptree_node_linkedlist = row["linked_list"]

            # send linked list, prefix and fptree_node
            fptree_node = fp_tree.add_prefix_to_fptree(prefix, fptree_node, fptree_node_linkedlist)

    # update count for each row in aux_ds
    for row in aux_ds:
        linked_list = row["linked_list"]
        updated_count = linked_list.get_count()
        row["count"] = updated_count

    # print the aux_ds
    for row in aux_ds:
        print(row)
        row["linked_list"].show_list()
        print("\n")

    fp_tree.visualize_fptree()

    ###### Step 4
    # find conditional database using suffix (use suffix to generate frequent patterns)
    # create conditional pattern base

    # create conditional database (cond_db)
    # cond_db = [{item: 85, cond_pattern_base: [], cond_fp_tree: [], freq_patterns_generated: []}]
    cond_db = []

    # initialize conditional database
    for row in aux_ds:
        temp_ds = {"item": row["id"], "cond_pattern_base": [], "cond_fp_tree": [], "freq_patterns_generated": []}
        cond_db.append(temp_ds)

    # start creating conditional pattern database using aux_ds 
    # starting with the bottom most item and then moving up

    # loop through aux_ds in reverse
    i = -1
    while i >= -len(aux_ds):
        print(aux_ds[i])

        # for each item in aux_ds, go to the linked list of that item
        item = aux_ds[i]["id"]
        current_linked_list = aux_ds[i]["linked_list"]

        # loop through linked list
        current_fptree_node = current_linked_list.head

        # get index of item in cond_db
        idx = [i for i, d in enumerate(cond_db) if item == d["item"]]

        # temp list for all conditional fp trees from an item
        all_cond_fptrees = []

        while current_fptree_node:
            # at each fp_node, loop through the parent of every
            # fp_node above it, and create a conditional pattern base
            conditional_pattern_base = find_conditional_pattern_base(current_fptree_node.parent)
            print(conditional_pattern_base)

            # use conditional pattern base to create conditional fp_tree
            cond_fp_tree = create_fp_tree_from_conditional_pattern_base(conditional_pattern_base)
            # cond_fp_tree.visualize_fptree()

            # add the conditional pattern base to the conditional db of respective item
            cond_db[idx[0]]["cond_pattern_base"].append(conditional_pattern_base)

            # add conditional fp tree to all_cond_fptrees
            all_cond_fptrees.append(cond_fp_tree)

            current_fptree_node = current_fptree_node.next

        # update loop
        i -= 1

        # join all the conditional fp tree
        joined_conditional_fptree = join_conditional_fptrees(all_cond_fptrees)
        all_cond_fptrees = []
        joined_conditional_fptree.visualize_fptree()

        # items with count in the joined conditional fp tree < min_sup is deleted
        pruned_conditional_fptree = prune_conditional_fptree(joined_conditional_fptree, min_sup)
        print('\n')

        # add pruned conditional fp tree to conditional db of respective item
        cond_db[idx[0]]["cond_fp_tree"] = pruned_conditional_fptree

        # Now, generate subsets of the joined and pruned conditional fp tree 
        item = cond_db[idx[0]]["item"]
        subsets = generate_subsets(item, pruned_conditional_fptree)

        # add generated subsets to frequent_patterns_generated of respective
        # item's conditional db
        cond_db[idx[0]]["freq_patterns_generated"] = subsets

        print(subsets)
        print('\n')

    # print cond_db
    for row in cond_db:
        print(row)

    # print all freq_patterns_generated of each row
    # and output them 
    for row in cond_db:
        freq_patterns_generated = row["freq_patterns_generated"]
        item = row["item"]

        for patterns in freq_patterns_generated:
            for pattern in patterns:
                output_pattern(pattern, item, dataset_name)

    return


def main():
    # start time
    start = time.time()

    # taking minimum support from sys.argv
    min_sup = int(sys.argv[1])
    # taking dataset name from sys.argv
    dataset_name = sys.argv[2]

    # load the dataset
    data = load_dataset(dataset_name)

    print_summary(data)

    # Perform Apriori Algorithm
    fp_growth(data, min_sup, dataset_name)

    # end time
    print('Total Elapsed Time: %s' % (time.time() - start))
    return


if __name__ == "__main__":
    main()
