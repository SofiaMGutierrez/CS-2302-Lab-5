#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 3 19:33:42 2019
Course: CS 2302
Author: Sofia Gutierrez
Lab #5: 
Instructor: Olac Fuentes
T.A.: Anindita Nath
"""
import numpy as np
import time

class WordEmbedding(object): 
    def __init__(self,word,embedding=[]): 
        # word must be a string, embedding can be a list or and array of ints or floats 
        self.word = word 
        self.emb = np.array(embedding, dtype=np.float32) 
        # For Lab 4, len(embedding=50)

########################### BST ###########################

class BST(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

def InsertBST(T,newItem):
    if T == None:
        T =  BST(newItem)
    elif T.data.word > newItem.word:
        T.left = InsertBST(T.left, newItem)
    else:
        T.right = InsertBST(T.right, newItem)
    return T

def HeightBST(T):
    if T == None:
        return 0
    lh = HeightBST(T.left)
    rh = HeightBST(T.right)
    return max([lh,rh])+1

def NumOfNodesBST(T):
    if T == None:
        return 0
    else:
        count = 1
        count += NumOfNodesBST(T.left) + NumOfNodesBST(T.right)
    return count

def SearchBST(T, k):
    if T == None:
        return None
    elif k == T.data.word:
        return T
    elif k < T.data.word:
        return SearchBST(T.left, k)
    else:
        return SearchBST(T.right, k)

########################### BTree ###########################

class BTree(object):
    # Constructor
    def __init__(self,data,child=[],isLeaf=True,max_data=5):  
        self.data = data
        self.child = child 
        self.isLeaf = isLeaf
        if max_data <3: #max_data must be odd and greater or equal to 3
            max_data = 3
        if max_data%2 == 0: #max_data must be odd and greater or equal to 3
            max_data +=1
        self.max_data = max_data

def FindChild(T,k):
    # Determines value of c, such that k must be in subtree T.child[c], if k is in the BTree   
    for i in range(len(T.data)):
        if k.word < T.data[i].word:
            return i
    return len(T.data)

def InsertInternal(T, i):
    # T cannot be Full
    if T.isLeaf:
        InsertLeaf(T,i)
    else:
        k = FindChild(T,i)   
        if IsFull(T.child[k]):
            m, l, r = Split(T.child[k])
            T.data.insert(k,m) 
            T.child[k] = l
            T.child.insert(k+1,r) 
            k = FindChild(T,i)  
        InsertInternal(T.child[k],i)

def Split(T):
    #print('Splitting')
    #PrintNode(T)
    mid = T.max_data//2
    if T.isLeaf:
        leftChild = BTree(T.data[:mid],max_data=T.max_data) 
        rightChild = BTree(T.data[mid+1:],max_data=T.max_data) 
    else:
        leftChild = BTree(T.data[:mid],T.child[:mid+1],T.isLeaf,max_data=T.max_data) 
        rightChild = BTree(T.data[mid+1:],T.child[mid+1:],T.isLeaf,max_data=T.max_data) 
    return T.data[mid], leftChild,  rightChild

def InsertLeaf(T,i):
    T.data.append(i)
    T.data.sort(key = lambda x: x.word)

def IsFull(T):
    return len(T.data) >= T.max_data

def Insert(T,i):
    if not IsFull(T):
        InsertInternal(T,i)
    else:
        m, l, r = Split(T)
        T.data =[m]
        T.child = [l,r]
        T.isLeaf = False
        k = FindChild(T,i)
        InsertInternal(T.child[k],i)

def HeightBTree(T):
    if T.isLeaf:
        return 0
    return 1 + HeightBTree(T.child[0])

def NumOfNodesBTree(T):
    sum = len(T.data)
    for i in T.child:
        sum+=NumOfNodesBTree(i)
    return sum

def SearchBTree(T,k):
    for i in range(len(T.data)):
        if k.word == T.data[i].word:
            return T.data[i]
    if T.isLeaf: 
        return None
    return SearchBTree(T.child[FindChild(T,k)],k)

########################### Hash tables with chaining ###########################

class HashTableChain(object):
    # Builds a hash table of size 'size'
    # Item is a list of (initially empty) lists
    # Constructor
    def __init__(self,size):
        self.bucket = [[] for i in range(size)] #makes the list of buckets

    #looks up where the item should be placed
    def h(self,k):
        return len(k.word)%len(self.bucket) #return the index that correspondes to the item

    def insert(self,k):
        # Inserts k in appropriate bucket (list)
        # Does nothing if k is already in the table
        b = self.h(k) #b is the index which correspondes to the item
        if not k.word in self.bucket[b]:
            self.bucket[b].append(k) #Insert new item at the end

    def find(self,k):
        # Returns bucket (b) and index (i)
        # If k is not in table, i == -1
        b = self.h(k)
        try:
            i = self.bucket[b].index(k) #if this gives an error it will continue on to "except"
        except:
            i = -1
        return b, i

    def print_table(self):
        print('Table contents:')
        for b in self.bucket:
            print(b)

    def string_lengthHTC(self,k):
        return len(k)%len(self.bucket)

    def asciiHTC(self,k):
        return ord(k[0])%len(self.bucket)

    def product_asciiHTC(self,k):
        return (ord(k[0])*ord(k[-1]))%len(self.bucket)

    def sum_asciiHTC(self,k):
        sum1 = 0
        for c in k:
            sum1 += ord(c)
        return sum1%len(self.bucket)

    def recursiveHTC(self,k):
        if len(k) == 0:
            return 1
        return (ord(k[0]) + 255 * self.h_recursive(k[1:]))%len(self.bucket)

########################### Hash tables with linear probing ###########################

class HashTableLP(object):
    # Builds a hash table of size 'size', initilizes items to -1 (which means empty)
    # Constructor
    def __init__(self,size):
        self.item = [WordEmbedding('-1') for x in range(size)]
    
    def hHtlp(self,k, i):
        return ((len(k)+i)%len(self.item))
    
    def insertHtlp(self,k):
        # Inserts k in table unless table is full
        # Returns the position of k in self, or -1 if k could not be inserted
        for i in range(len(self.item)): #Despite for loop, running time should be constant for table with low load factor
            pos = self.hHtlp(k.word, i)
            if self.item[pos].word == '-1' or self.item[pos].word == '2':
                self.item[pos] = k
                return pos
        return -1

    def findHtlp(self,k):
        # Returns the position of k in table, or -1 if k is not in the table
        for i in range(len(self.item)):
            pos = self.hHtlp(k.word, i)
            if self.item[pos].word == k.word:
                return pos
            if self.item[pos] == '-1':
                return -1

    def length_stringHTLP(self,k, i):
        return ((len(k)+i)%len(self.item))

    def asciiHTLP(self,k, i):
        return (ord(k[0])+i)%len(self.item)

    def product_asciiHTLP(self,k, i):
        return ((ord(k[0])*ord(k[-1]))+i)%len(self.item)

    def sum_asciiHTLP(self,k, i):
        sum1 = i
        for c in k:
            sum1 += ord(c)
        return sum1%len(self.item)

    def recursiveHTLP(self,k, i):
        if len(k) == 0:
            return 1
        return ((ord(k[0]) + 255 * self.h_recursive(k[1:], i)) + i)%len(self.item)
    
if __name__ == "__main__":
    
    print("1: Binary search tree")
    print("2: B-tree")
    print("3: Hash tables with chaining")
    print("4: Hash tables with linear probing")
    menu_choice = int(input("Enter a menu option: "))
    
########################### BST ###########################
    if menu_choice == 1:
        
        print("Building binary search tree...")
        BST_T = None
        
        with open("glove.6B.50d.txt", "r", encoding='utf-8') as file:
            start1 = time.time()
            for line in file:
                line = line.split(" ")
                word_object = WordEmbedding(line[0], line[1:])
                BST_T = InsertBST(BST_T, word_object)
        end1 = time.time()
        
        print("Binary Search Tree stats:")
        print("Number of nodes:", NumOfNodesBST(BST_T))
        print("Height:", HeightBST(BST_T))
        print("Running time for binary search tree construction:", end1-start1)
        
        with open("words.txt", "r") as file: 
            start2 = time.time()
            for line in file:
                line = line.strip().split(" ")
                word1 = SearchBST(BST_T, line[0])
                word2 = SearchBST(BST_T, line[1])
                
                distance = np.dot(word1.data.emb,word2.data.emb)/(abs(np.linalg.norm(word1.data.emb))*abs(np.linalg.norm(word2.data.emb)))
                print("Similarity [", word1.data.word, word2.data.word, "] =", distance)
                
        end2 = time.time()
        
        print("Running time for binary search tree query processing:", end2-start2)

########################### BTree ###########################
    if menu_choice == 2:
        
        user_max_data = int(input("Maximum number of items in node: "))
        print("Building B-tree...")
        BTree_T = BTree([], max_data = user_max_data)
        
        with open("glove.6B.50d.txt", "r", encoding='utf-8') as file:
            start1 = time.time()
            for line in file:
                list1 = line.split(" ")
                word_object = WordEmbedding(list1[0],list1[1:])
                Insert(BTree_T, word_object)
        end1 = time.time()
        
        print("B-tree stats:")
        print("Number of nodes:", NumOfNodesBTree(BTree_T))
        print("Height:", HeightBTree(BTree_T))
        print("Running time for B-tree construction:", end1-start1)
        
        with open("words.txt", "r") as file:
            start2 = time.time()
            for line in file:
                line = line.strip().split(" ")
                
                obj1 = WordEmbedding(line[0])
                obj2 = WordEmbedding(line[1])
                
                word1 = SearchBTree(BTree_T, obj1)
                word2 = SearchBTree(BTree_T, obj2)
                
                distance = np.dot(word1.emb,word2.emb)/(abs(np.linalg.norm(word1.emb))*abs(np.linalg.norm(word2.emb)))
                print("Similarity [", word1.word, word2.word, "] =", distance)
                
        end2 = time.time()
        
        print("Running time for B-tree query processing:", end2-start2)

########################### Hash tables with chaining ###########################
    if menu_choice == 3:

        print("Building hash table with chaining...")
        htc = HashTableChain(3)
        
        with open("glove.6A.50d.txt", "r", encoding='utf-8') as file:
            start1 = time.time()
            for line in file:
                list1 = line.split(" ")
                word_object = WordEmbedding(list1[0],list1[1:])
                htc.insert(word_object)
        end1 = time.time()
        print("Running time for hash table with chaining construction:", end1-start1)

        with open("words.txt", "r") as file:
            start2 = time.time()
            for line in file:
                line = line.strip().split(" ")
                
                obj1 = WordEmbedding(line[0])
                obj2 = WordEmbedding(line[1])
                
                bucket1, index1 = htc.find(obj1)
                bucket2, index2 = htc.find(obj2)
                
                distance = np.dot(htc.bucket[bucket1][index1].emb,htc.bucket[bucket2][index2].emb)/(abs(np.linalg.norm(htc.bucket[bucket1][index1].emb))*abs(np.linalg.norm(htc.bucket[bucket2][index2].emb)))
                print("Similarity [", htc.bucket[bucket1][index1].word, htc.bucket[bucket2][index2].word, "] =", distance)

        end2 = time.time()

        print("Running time for hash table with chaining query processing:", end2-start2)

########################### Hash tables with linear probing ###########################
    if menu_choice == 4:
        print("Building has table with linear probing...")
        
        #the following is to initialize the size of the hash table since every item has to be in its own bucket
        counter = 0
        with open("glove.6A.50d.txt", "r", encoding='utf-8') as file:
            for line in file:
                counter += 1
        htlp = HashTableLP(counter)
    
        with open("glove.6A.50d.txt", "r", encoding='utf-8') as file:
            start1 = time.time()
            for line in file:
                list1 = line.split(" ")
                word_object = WordEmbedding(list1[0],list1[1:])
                htlp.insertHtlp(word_object)
        end1 = time.time()
        print("Running time for hash table with linear probing construction:", end1-start1)
    
        with open("words.txt", "r") as file:
            start2 = time.time()
            for line in file:
                line = line.strip().split(" ")
                
                obj1 = WordEmbedding(line[0])
                obj2 = WordEmbedding(line[1])
                
                item1 = htlp.findHtlp(obj1)
                item2 = htlp.findHtlp(obj2)

                distance = np.dot(htlp.item[item1].emb,htlp.item[item2].emb)/(abs(np.linalg.norm(htlp.item[item1].emb))*abs(np.linalg.norm(htlp.item[item2].emb)))
                print("Similarity [", htlp.item[item1].word, htlp.item[item2].word, "] =", distance)
    
        end2 = time.time()
    
        print("Running time for hash table with linear probing query processing:", end2-start2)