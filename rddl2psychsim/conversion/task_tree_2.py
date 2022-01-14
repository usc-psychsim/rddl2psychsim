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
from psychsim.pwl.tree import KeyedTree

def extract_from_actionset(actset):    
    a1lst = [a for a in actset][0]
    player = a1lst['subject']
    action = a1lst['verb']
    a_name = player + ':' + action
    return player, a_name

class TreeNode:
    def __init__(self, name, ntype, npsim, nplayer=None):
        self.tree = None
        self.children = []
        self.parents = []
        self.name = name
        self.type = ntype
        self.value = None
        self.psim_name = npsim
        self.player = nplayer
        
    def is_root(self):
        return len(self.parents) == 0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def add_parent(self, par):
        if par.name in [self.name] + self.children:
            print('ERROR add parent', par.name, 'to', self.name, 'with children', self.children)
            return
        
        self.parents.append(par.name)
        
    def add_child(self, ch):
        if ch.name in [self.name] + self.parents:
            print('ERROR add child', ch.name, 'to', self.name, 'with parents', self.parents)
            return
        
        self.children.append(ch.name)
    
    def __str__(self, level=0):
        ret = '\n' + "\t"*level + self.psim_name 
        if self.value is not None:
            if type(self.value) == Distribution:
                self.value = self.value.first()
            ret += '=' + str(self.value) # + str(type(self.value))
        for child in self.children:
            ret += self.tree.nodes_dict[child].__str__(level+1)
        return ret       
        

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
    
    def add_node(self, node):
        self.nodes_dict[node.name] = node
        node.tree = self
    
    def add_edge_by_force(self, parent, child):        
        if child.name in self.nodes_dict.keys():
            child_node = self.nodes_dict[child.name]
        else:
            child_node = deepcopy(child)
        if parent.name in self.nodes_dict.keys():
            parent_node  = self.nodes_dict[parent.name]
        else:
            parent_node = deepcopy(parent)
        self.add_node(parent_node)
        self.add_node(child_node)
        parent_node.add_child(child_node)
        child_node.add_parent(parent_node)
        
        # if this child was root, remove from roots
        if child.name in self.roots:
            self.roots.remove(child.name)
        if parent_node.is_root():
            self.roots.add(parent.name)
            
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
            self.add_node(o_node)
            
                      
        ## if creating an edge between affected and affecting
        if affecting is not None:
            self.add_edge(affected, affecting)
        
        affected_here = self.nodes_dict[affected.name]
        if affected_here.is_leaf():
            affected_there = other_tree.nodes_dict[affected.name]
            for ch in affected_there.children:
                affected_here.add_child(self.nodes_dict[ch])
                self.nodes_dict[ch].add_parent(affected_here)
        
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
    
    def get_blockers_of_node(self, node_name, level=0):
        node = self.nodes_dict[node_name]
        if node.value == True:
            return {}
        elif (node.value == False):
            if node.is_leaf():
                if node.type == ACTION:
                    return {(level, node_name)}
                else:
                    return {}
            ## If I'm false, find out blockers of my children            
            blockers = set()
            for child in node.children:
                blockers = blockers.union(self.get_blockers_of_node(child.name, level+1))            
            return blockers
        else:
            print('Value problem', node.value)
            return None
        
    
    def get_blockers_of_node2(self, node_name, level=0, chain=[]):
        node = self.nodes_dict[node_name]
        if node.value == True:
            return [chain]
        
        ## If I'm false, find out blockers of my children
        elif (node.value == False):
            new_chain = list(chain)            
            new_chain.append((level, node))

            if node.is_leaf():
                return [new_chain]
                
            blockers = []
            for child in node.children:
                blockers = blockers + self.get_blockers_of_node2(child, level+1, new_chain)
            return blockers
        else:
            print('Value problem', node.value)
            return [chain]
    
    def get_blockers(self):
        blockers = {root:self.get_blockers_of_node2(root, level=0, chain=[]) for root in self.roots}
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

def chain_rep(chain):
    return [(n[0], n[1].name) for n in chain]

def add_chain(ochain, ochains):
    chain = chain_rep(ochain)
    chains = [chain_rep(c) for c in ochains]
    if chain in chains:
        return False, ochains
    remove = []
    for och in ochains:
        if len(och) > len(ochain):
            # if chain is a subset of an existing chain, we're done
            if chain_rep(och)[:len(chain)] == chain:
                return False, ochains
        else:
            # if existing chain is subset of this chain, remove existing
            if chain[:len(och)] == chain_rep(och):
                remove.append(chain_rep(och))
                
    new_chains = [ch for ch in ochains if chain_rep(ch) not in remove]
    new_chains.append(ochain)
    return True, new_chains

class AllTrees:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.dyn_trees = dict()
        self.legal_trees = dict()   # unused for now
        self.nodes_dict = {}
        self.edges = []
        self.reward_trees = set()
        self.top_names = []
        self.non_boolean_names = {}
        
    def add_toplevel_names(self, name):
        self.top_names.append(name)
        for it, tree in self.dyn_trees.items():
            for root in tree.roots:
                if name in root:
                    self.reward_trees.add(it)
                    
    def who_did_what(self, actions):
        action_node_names = {extract_from_actionset(a)[1]:extract_from_actionset(a)[0] for a in actions}
        player2trees = {p:[] for p in action_node_names.values()}
        for it in self.reward_trees:
            tree = self.dyn_trees[it]
            for node in tree.nodes_dict.values():
                if node.psim_name in action_node_names.keys():
                    player2trees[action_node_names[node.psim_name]].append(it)
        return player2trees
        
        
    def get_blockers(self):
        for tree in self.dyn_trees.keys():
            blks = self.dyn_trees[tree].get_blockers()
            blks = {k:v for k,v in blks.items() if 'saved' in k}
            print('tttttt', tree)
            for root, blks in blks.items():
                print('\trrrrrr', root)
                seen_chains = []
                for chain in blks:
                    if len(chain) == 1:
                        continue
                    chain_added, seen_chains = add_chain(chain, seen_chains)
                    if not chain_added:
                        continue
                    print('\t\tccccccc', chain_rep(chain))
                    
                    # Take the leaf of this chain and check if it appears as a parent in any tree
                    leaf = chain[-1][1]
                    print('\t\t\tddddd', leaf.name)
                    for t in self.dyn_trees.values():
                        if t.id == tree:
                            continue
                        if leaf.name not in t.nodes_dict.keys():
                            continue
                        deeper_chains = t.get_blockers_of_node2(leaf.name, level=chain[-1][0], chain=chain[:-1])
                        for d_chain in deeper_chains:
                            d_chain_added, seen_chains = add_chain(d_chain, seen_chains)
                            if not d_chain_added:
                                continue
                            print('\t\t\t>>>', chain_rep(d_chain))
#                            print('***** Collapse', tree, t.id)
                print('\tfffff', [chain_rep(c) for c in seen_chains])
            
        
    def create_node(self, name, ntype, psim_name, player=None):
        if name in self.nodes_dict.keys():
            return
        self.nodes_dict[name] = TreeNode(name, ntype, psim_name, player)
        
    def make_tree(self, affected, affecting, is_legality_tree=False):
        # Create a new tree
        new_tree = TaskTree(len(self.dyn_trees), self.verbose)            
        new_tree.add_edge_by_force(affected, affecting)
        if self.verbose: print("New tree:", affected.name, affecting.name, 'id', new_tree.id) 
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
            if self.verbose: print('Attaching to existing', n_affecting, 'tree', affing_tree.id)
            affing_tree.add_edge(affected, affecting)
            print(affing_tree)
             
        elif (affed_tree is not None):
            if self.verbose: print('Attaching to existing', n_affected, 'tree', affed_tree.id)
            affed_tree.add_edge(affected, affecting)
            print(affed_tree)
            
        self.edges.append((n_affected, n_affecting))
        
    
    def attach_multi(self, n_affected, n_affecting):
        if (n_affected, n_affecting) in self.edges:
            print('Already added', n_affected, n_affecting)
            return
        affected = self.nodes_dict[n_affected]
        affecting = self.nodes_dict[n_affecting]
        affing_trees = []
        affed_trees = []
        for tree in self.dyn_trees.values():
            if tree.contains(n_affected):
                affed_trees.append(tree.id)
            elif tree.contains(n_affecting):
                affing_trees.append(tree.id)

        if (affing_trees == []) and (affed_trees == []):
            self.make_tree(affected, affecting)
            
        elif (len(affing_trees) > 0) and (len(affed_trees) > 0):
            if self.verbose: print('Collapsing', affed_trees[0], 'subsumes', affing_trees[0])
            self.dyn_trees[affed_trees[0]].subsume(affected, affecting, self.dyn_trees[affing_trees[0]])
            del self.dyn_trees[affing_trees[0]]
#            for t in affing_trees[1:]:
#                t.add_edge(affected, affecting)
#            for t in affed_trees[1:]:
#                t.add_edge(affected, affecting)
            
        elif len(affing_trees) > 0:
            if self.verbose: print('Attaching to existing', n_affecting)
            for t in [self.dyn_trees[tr] for tr in affing_trees]:
                t.add_edge(affected, affecting)
#            self.print()
             
        elif len(affed_trees) > 0:
            if self.verbose: print('Attaching to existing', n_affected)
            for t in [self.dyn_trees[tr] for tr in affed_trees]:
                t.add_edge(affected, affecting)
#            self.print()
            
        self.edges.append((n_affected, n_affecting))
                
    def print(self, reward_trees_only=False):
        if reward_trees_only:
            ids = self.reward_trees
        else:
            ids = self.dyn_trees.keys()
        for i in ids:
            print(i, self.dyn_trees[i])

    def final_stitch(self):
        ## Identify trees candidates for remove
        cands = [tree for tree in self.dyn_trees.values() if (len(tree.roots)==1) and (tree.id not in self.reward_trees)]
        for cand in cands:
            cand_root = list(cand.roots)[0]
            for rew_t in self.reward_trees:
                rew_tree = self.dyn_trees[rew_t]
                if (cand_root in rew_tree.nodes_dict.keys()) and rew_tree.nodes_dict[cand_root].is_leaf():
                    rew_tree.subsume(cand.nodes_dict[cand_root], None, cand)
                    print('Attached', cand.id, 'to', rew_t)
                    del self.dyn_trees[cand.id]
                    
    def get_node(self, nname):
#        fluent_name = 
        pass
            
    def extract_tests(self, tree):
        var_2_val = dict()
        var_2_branches = dict()
        bools = []
        branch_plane = tree.branch
        if branch_plane is None:
            return var_2_val, bools
        for kv, thresh, comp in branch_plane.planes:
            if len(kv._data) > 1:
                print('ERROR Dont know what to do', tree)
                return None
            pvar = list(kv._data.keys())[0]
            is_non_bool = pvar in self.non_boolean_names.values()
            if is_non_bool:
                if comp == 0:
                    var_2_val[pvar] = thresh
                else:
                    if pvar not in var_2_branches:
                        var_2_branches[pvar] = []
                    var_2_branches[pvar].append((thresh, comp))
            else:
                bools.append(pvar)
        
        ## For non-booleans that have multipleb branches
        for nonb, branches in var_2_branches.items():
            if len(branches) != 2:
                print('ERROR Dont know what to do', tree)
                return None
            ## If same threshold and opposite comparisons
            threshs = set([branches[0][0], branches[1][0]])
            comps =   set([branches[0][1], branches[1][1]])
            if (len(threshs) == 1) and (len(comps) == 2):
                var_2_val[nonb] = threshs.pop() 
            
        return var_2_val, bools
    
    def build(self, dynamics, legality):        
        for key, dyn_dict in dynamics.items():
            ## Action dynamics 
            if type(key) == ActionSet:
                player, a_name = extract_from_actionset(key)
                
                ## Add the action node
                self.create_node(a_name, ACTION, a_name, player)
                                
                ## Dynamics tree
                ## Fluents in this tree are affected by the action
                for affected_psim_name in dyn_dict.keys():
                    affected_name = clean_str(affected_psim_name)
                    print('Dyn:', a_name, 'affects', affected_name )
                    self.attach_multi(affected_name , a_name)
                    
                ## Legality tree
                ## Fluents in this tree affect the action
                legality_tree = legality[player].get(key, dict())
                if type(legality_tree) != dict:
                    var_2_val, bools = self.extract_tests(legality_tree)
                    print(var_2_val, bools, legality_tree)
                affecting_fluents = list(legality_tree.keys())
                new_tree = None
                parent_node = self.nodes_dict[a_name]
                for affecting_psim_name in affecting_fluents:
                    affecting_name = clean_str(affecting_psim_name)
                    if affecting_name not in self.nodes_dict:
                        print('hi')
                    affecting_node = self.nodes_dict[affecting_name]
                    print('Legality:', a_name, 'affected by', affecting_name)
                    if new_tree is None:
                        new_tree = self.make_tree(parent_node, affecting_node) #, is_legality_tree=True
                    else:
                        new_tree.add_edge(parent_node , affecting_node )
                    
            ## Fluent dynamics 
            if type(key) == str:        
                affected_name = clean_str(key)
                print('Fluent:', affected_name, 'affected by list', dyn_dict.keys())
                if 'triaged' in affected_name:
                    print('hi')
                for action_or_true, dtree in dyn_dict.items():
                    # Ignore action_or_true because the action dependency was captured above
                    affecting = clean_strs(dtree.keys(), affected_name) 
                    for affing in affecting:  
                        print('\taffected by', affing)
                        if 'ACTION' in affing:
                            print('hi')
                        self.attach_multi(affected_name, affing)                    
        
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
    