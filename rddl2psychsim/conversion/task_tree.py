#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 12:07:13 2021

@author: mostafh
"""

ACTION = 0
KNOW_PROP = 1
PROP = 2
PLAYERS = ['Eng', 'Med', 'Tran']

from copy import deepcopy
from psychsim.action import ActionSet
from psychsim.probability import Distribution

def extract_from_actionset(actset):    
    a1lst = [a for a in actset][0]
    player = a1lst['subject']
    action = a1lst['verb']
    a_name = player + ':' + action
    return player, a_name

class TreeNode:
    def __init__(self, name, ntype, npsim, nvalue=None, nplayer=None):
        self.children = []
        self.parents = []
        self.name = name
        self.type = ntype
        self.value = nvalue
        self.psim_name = npsim
        self.player = nplayer
        
    def is_root(self):
        return len(self.parents) == 0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def add_parent(self, par):
        if par.name in [self.name] + [ch.name for ch in self.children]:
            print('ERROR add parent', par.name, 'to', self.name, 'with children', [c.name for c in self.children])
            return
        
        self.parents.append(par)
        
    def add_child(self, ch):
        if ch.name in [self.name] + [par.name for par in self.parents]:
            print('ERROR add child', ch.name, 'to', self.name, 'with parents', [p.name for p in self.parents])
            return
        
        self.children.append(ch)
    
    def __str__(self, level=0):
        ret = '\n' + "\t"*level + self.psim_name 
        if self.value is not None:
            if type(self.value) == Distribution:
                self.value = self.value.first()
            ret += '=' + str(self.value) # + str(type(self.value))
        for child in self.children:
            ret += child.__str__(level+1)
        return ret       
        
#    def get_player(self):
#        if self.type != ACTION:
#            return None
#
#        
#        ## If a player-specific action
#        ## Get player name from action name
#        if '_' in self.name:
#            start = 0
#            if '(' in self.name:
#                start = 1
#            return self.name[start:self.name.index('_')]
#    
#        ## If action can be done by anyone, player is the first arg of the action
#        args = [arg.strip() for arg in self.name.split(',')]
#        ## If first arg is a player name, return it
#        if args[1] in PLAYERS:
#            return args[1]
#                
#        print("Error getting player from", self.name)
#        return None
        

class TaskTree:    
    def __init__(self, id, verbose=False):
        self.id = id
        self.roots = set()
        self.verbose = verbose
        self.nodes_dict = {}
        self.player_2_action_nodes = {}
        
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
        parent_node.add_child(child_node)
        child_node.add_parent(parent_node)
        
        # if this child was root, remove from roots
        if child.name in self.roots:
            self.roots.remove(child.name)
        if parent_node.is_root():
            self.roots.add(parent_node.name)
            
        ## If child is action (actions can't be parents; they're not affected)
        ## extract the player who's doing it
        if child_node.type == ACTION:
            player = child_node.player
            if player not in self.player_2_action_nodes:
                self.player_2_action_nodes[player] = set()
            self.player_2_action_nodes[player].add(child_node.name)
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
                
        ## Add the player 2 action nodes mapping
        for player, node_names in other_tree.player_2_action_nodes.items():
            if player not in self.player_2_action_nodes:
                self.player_2_action_nodes[player] = set()
            self.player_2_action_nodes[player] = self.player_2_action_nodes[player].union(other_tree.player_2_action_nodes[player])
            
    
    def __str__(self):
        s = ''
        for rt in self.roots:
            s += str(self.nodes_dict[rt])
        return s
    
    def get_blockers_of_node(self, node_name):
        node = self.nodes_dict[node_name]
        if node.value == True:
            return {}
        elif (node.value == False):
            if node.is_leaf():
                return {node_name}
            ## If I'm false, find out blockers of my children            
            blockers = set()
            for child in node.children:
                blockers = blockers.union(self.get_blockers_of_node(child.name))            
            return blockers
        else:
            print('Value problem', node.value)
            return None
    
    def get_blockers(self):
        blockers = {root:self.get_blockers_of_node(root) for root in self.roots}
        return blockers
    
    def eval_from_world(self, world, actions):
        action_node_names = [extract_from_actionset(a)[1] for a in actions]
        for name, node in self.nodes_dict.items():
            if node.type == PROP:
                node.value = world.getFeature(node.psim_name, unique=True)
            if node.type == ACTION:
                node.value = (node.psim_name in action_node_names)
           
def clean_str(ps):
    if ps == '':
        return ps
    if ps[-1] == '\'':
        return ps[:-1]
    return ps

def clean_strs(psim_strs, rmv):
    ret = []
    for ps in psim_strs:
        cps = clean_str(ps)
        if (len(cps) == 0) or (cps == rmv):
            continue
        ret.append(cps)
    return ret

class AllTrees:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.dyn_trees = dict()
        self.legal_trees = dict()
        self.nodes_dict = {}
        self.edges = []
        
    def create_node(self, name, ntype, psim_name, value=None, player=None):
        if name in self.nodes_dict.keys():
            return
        self.nodes_dict[name] = TreeNode(name, ntype, psim_name, value, player)
        
    def make_tree(self, affected, affecting, is_legality_tree=False):
        # Create a new tree
        new_tree = TaskTree(len(self.dyn_trees), self.verbose)            
        new_tree.add_edge_by_force(affected, affecting)
        if self.verbose: print("New tree:", affected.name, affecting.name) 
        if is_legality_tree:
            self.legal_trees[affecting.name] = new_tree            
        else:
            self.dyn_trees[new_tree.id] = new_tree
        return new_tree
    
    def copy_world_values(self, world, actions):
        for tree in self.dyn_trees.values():
            tree.eval_from_world(world, actions)
    
    def attach(self, n_affected, n_affecting):
        if (n_affected, n_affecting) in self.edges:
            print('Already added', n_affected, n_affecting)
            return
        affected = self.nodes_dict[n_affected]
        affecting = self.nodes_dict[n_affecting]
        affing_tree = None
        affed_tree = None
        for tree in self.dyn_trees.values():
            if tree.contains(n_affected):
                affed_tree = tree
            elif tree.contains(n_affecting):
                affing_tree = tree

        if (affing_tree is None) and (affed_tree is None):
            self.make_tree(affected, affecting)
            
        elif (affing_tree is not None) and (affed_tree is not None):
            if self.verbose: print('Collapsing', affed_tree.id, 'subsumes', affing_tree.id)
            affed_tree.subsume(affected, affecting, affing_tree)
            del self.dyn_trees[affing_tree.id]
            
        elif (affing_tree is not None):
            if self.verbose: print('Attaching to existing', n_affecting)
            affing_tree.add_edge(affected, affecting)
             
        elif (affed_tree is not None):
            if self.verbose: print('Attaching to existing', n_affected)
            affed_tree.add_edge(affected, affecting)
            
        self.edges.append((n_affected, n_affecting))
                
    def print(self):
        for i, tree in self.dyn_trees.items():
            print(i, tree)
        for i, tree in self.legal_trees.items():
            print(i, tree)

            
    def build(self, dynamics, legality):        
        for key, dyn_dict in dynamics.items():
            ## If action dynamics 
            if type(key) == ActionSet:
                player, a_name = extract_from_actionset(key)
                
                ## Add the action node
                self.create_node(a_name, ACTION, a_name, None, player)
                
                ## Get the legality tree for this agent for this action
                legality_tree = legality[player].get(key, dict())                
                ## Get the fluents in the tree. These affect the action
                affecting_fluents = list(legality_tree.keys())
                new_tree = None
                parent_node = self.nodes_dict[a_name]
                for affecting_psim_name in affecting_fluents:
                    affecting_name = clean_str(affecting_psim_name)
                    affecting_node = self.nodes_dict[affecting_name]
                    print('Legality:', a_name, 'affected by', affecting_name)
                    if new_tree is None:
                        new_tree = self.make_tree(parent_node, affecting_node, is_legality_tree=True)
                    else:
                        new_tree.add_edge(parent_node , affecting_node )
                
                
                ## Add an edge for the action-fluent dependency
                for affected_psim_name in dyn_dict.keys():
                    affected_name = clean_str(affected_psim_name)
                    print('Dyn:', a_name, 'affects', affected_name )
                    self.attach(affected_name , a_name)
                    
            ## If fluent dynamics 
            if type(key) == str:        
                affected_name = clean_str(key)
                for tkey in dyn_dict.keys():
                    affecting = clean_strs(dyn_dict[tkey].keys(), affected_name) 
                    for affing in affecting:  
                        print('Fluent:', affected_name, 'affected by', affing)
                        if 'ACTION' in affing:
                            print('hi')
                        self.attach(affected_name, affing)
                        
    def eval_from_world(self, world):
        for tree in self.dyn_trees.values():
            tree.eval_from_world(world)
            
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
    