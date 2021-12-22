#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 12:07:13 2021

@author: mostafh
"""

ACTION = 0
KNOW_PROP = 1
PROP = 2

from copy import deepcopy
import numpy as np

class TreeNode:
    def __init__(self, name, ntype, npsim, nvalue=None):
        self.children = []
        self.parents = []
        self.name = name
        self.type = ntype
        self.value = None
        self.value = nvalue
        self.psim_name = npsim
        
    def is_root(self):
        return len(self.parents) == 0
    
    def __str__(self, level=0):
        ret = '\n' + "\t"*level + self.name #+ ' parents ' + str([p.name for p in self.parents])
        for child in self.children:
            ret += child.__str__(level+1) # + ' parents ' + str([p.name for p in self.parents])
        return ret       
        

class TaskTree:    
    def __init__(self, id, verbose=False):
        self.id = id
        self.roots = set()
        self.verbose = verbose
        self.nodes_dict = {}
        
    def add_edge(self, parent, child):
        if not (self.contains(parent.name) or self.contains(child.name)):
                return False
        return self.add_edge_by_force(parent,child)
    
    def add_edge_by_force(self, parent, child):        
        if child.name in self.nodes_dict.keys():
            child_node = self.nodes_dict[child.name]
        else:
            child_node = deepcopy(child)
        if parent.name in self.nodes_dict.keys():
            parent_node  = self.nodes_dict[parent.name]
        else:
            parent_node = deepcopy(parent)
        self.nodes_dict[parent_node.name] = parent_node
        self.nodes_dict[child_node.name] = child_node
        parent_node.children.append(child_node)
        child_node.parents.append(parent_node)
        
        # if this child was root, make the parent the root
        if child.name in self.roots:
            self.roots.remove(child.name)
        if parent_node.is_root():
            self.roots.add(parent_node.name)
        return True

    def contains(self, node_name):
        return node_name in self.nodes_dict.keys()
    
    def subsume(self, affected, affecting, other_tree):
        ## Copy all the nodes in other tree except for the node you already have
        for o_node in other_tree.nodes_dict.values():
            if o_node.name == affected.name:
                continue
            self.nodes_dict[o_node.name] = o_node
                        
        ## Create an edge between affected and affecting
        self.add_edge(affected, affecting)
        
        ## Add the roots of the other tree is they're still parent-less
        for o_root in other_tree.roots:
            node_here = self.nodes_dict[o_root]
            if node_here.is_root():
                self.roots.add(node_here.name)
    
    def __str__(self):
        s = ''
        for rt in self.roots:
            s += str(self.nodes_dict[rt])
        return s
           
class AllTrees:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.trees = dict()
        self.nodes_dict = {}
        self.edges = []
        
    def create_node(self, name, ntype, psim_name, value=None):
        if name in self.nodes_dict.keys():
            return
        self.nodes_dict[name] = TreeNode(name, ntype, psim_name, value)
    
    def attach(self, n_affected, n_affecting):
        if (n_affected, n_affecting) in self.edges:
            print('Already added', n_affected, n_affecting)
            return
        affected = self.nodes_dict[n_affected]
        affecting = self.nodes_dict[n_affecting]
        affing_tree = None
        affed_tree = None
        for tree in self.trees.values():
            if tree.contains(n_affected):
                affed_tree = tree
            elif tree.contains(n_affecting):
                affing_tree = tree

        if (affing_tree is None) and (affed_tree is None):
            # Create a new tree
            new_tree = TaskTree(len(self.trees), self.verbose)            
            new_tree.add_edge_by_force(affected, affecting)
            if self.verbose: print("New tree:", affected.name, affecting.name) 
            self.trees[new_tree.id] = new_tree
            
        elif (affing_tree is not None) and (affed_tree is not None):
            if self.verbose:
                print('Collapsing', affed_tree.id, 'subsumes', affing_tree.id)
            affed_tree.subsume(affected, affecting, affing_tree)
            del self.trees[affing_tree.id]
            
        elif (affing_tree is not None):
            affing_tree.add_edge(affected, affecting)
             
        elif (affed_tree is not None):
            affed_tree.add_edge(affected, affecting)
            
        self.print()
        self.edges.append((n_affected, n_affecting))
            
        
            
                
    def print(self):
        for i, tree in self.trees.items():
            print(i, tree)
            
if __name__ == "__main__":
    at = AllTrees(True)
    at.create_node('save_v1', PROP, None)
    at.create_node('triaged_v1', PROP, None)
    at.create_node('evacd_v1', PROP, None)
    
    at.create_node('evacd_v1', PROP, None)
    at.create_node('transport_v1', ACTION, None)
    at.create_node('know_v1', KNOW_PROP, None)
    
    at.attach(['save_v1', 'triaged_v1', 'evacd_v1'])
    at.attach(['evacd_v1', 'transport_v1', 'know_v1'])
    
    at.create_node('smthg_else_v1', ACTION, None)
    at.attach(['smthg_else_v1', 'know_v1'])
    
    at.print()
    