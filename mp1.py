"""
Artificial Intelligence
MP1: A* for Sliding Puzzle
SEMESTER: FALL 2019
NAME: Jason Huggy
"""

import numpy as np
import queue

class PuzzleState():
    SOLVED_PUZZLE = np.arange(9).reshape((3, 3))

    def __init__(self,conf,g,predState):
        self.puzzle = conf     # Configuration of the state
        self.gcost = g         # Path cost - Should not be needed, because the distance to the next node will always be the same
        self._compute_heuristic_cost()  # Set heuristic cost
        self.fcost = self.gcost + self.hcost
        self.pred = predState  # Predecesor state
        self.zeroloc = np.argwhere(self.puzzle == 0)[0]
        self.action_from_pred = None
    
    def __hash__(self):
        return tuple(self.puzzle.ravel()).__hash__()
    
    def _compute_heuristic_cost(self):
        # Number of steps to quickest solution
        self.hcost = 0
        for i in self.puzzle: 
            for j in i:
                curr = []
                solve = []
                curr = np.asarray(np.where(self.puzzle == j))
                solve = np.asarray(np.where(PuzzleState.SOLVED_PUZZLE == j))
                x1 = curr.item(0)
                y1 = curr.item(1)
                x2 = solve.item(0)
                y2 = solve.item(1)
                self.hcost += abs(x1-x2) + abs(y1-y2)
    
    def is_goal(self):
        return np.array_equal(PuzzleState.SOLVED_PUZZLE,self.puzzle)
    
    def __eq__(self, other):
        return np.array_equal(self.puzzle, other.puzzle)
    
    def __lt__(self, other):
        return self.fcost < other.fcost
    
    def __str__(self):
        return np.str(self.puzzle)
    
    move = 0
    
    def show_path(self):
        if self.pred is not None:
            self.pred.show_path()
        
        if PuzzleState.move==0:
            print('START')
        else:
            print('Move',PuzzleState.move, 'ACTION:', self.action_from_pred)
        PuzzleState.move = PuzzleState.move + 1
        print(self)
    
    # Returns true if move is valid
    def can_move(self, direction):
        
        poss_moves = []
        if list(self.zeroloc) == [0, 0]:
            poss_moves = ['right','down']
        if list(self.zeroloc) == [0, 1]:
            poss_moves = ['right','left','down']
        if list(self.zeroloc) == [0, 2]:
            poss_moves = ['left','down']
        if list(self.zeroloc) == [1, 0]:
            poss_moves = ['up','right','down']
        if list(self.zeroloc) == [1, 1]:
            poss_moves = ['up','left','right','down']
        if list(self.zeroloc) == [1, 2]:
            poss_moves = ['up','left','down']
        if list(self.zeroloc) == [2, 0]:
            poss_moves = ['right','up']
        if list(self.zeroloc) == [2, 1]:
            poss_moves = ['right','left','up']
        if list(self.zeroloc) == [2, 2]:
            poss_moves = ['left','up'] 
        
        if direction in poss_moves:
            return True
        else:
            return False
        
    def gen_next_state(self, direction):
        
        # Saves the previous puzzle state
        puzz = []
        for i in self.puzzle:
            puzz.append(i) 
        puzzle = np.asarray(puzz)
        old = PuzzleState(puzzle, 0, self)
        old.action_from_pred = direction
        
        # Flattens puzzle and then finds index of neighbor of zero
        new = old.puzzle.ravel()
        zero_loc = []
        zero_loc = np.where(new == 0)
        
        if direction == 'up':
            swap_value_index = zero_loc[0] - 3
            
        if direction == 'down':
            swap_value_index = zero_loc[0] + 3
            
        if direction == 'left':
            swap_value_index = zero_loc[0] - 1
            
        if direction == 'right':
            swap_value_index = zero_loc[0] + 1
        
        # Swaps zero with the desired neighbor
        new[zero_loc] = new[swap_value_index]
        new[swap_value_index] = 0
        new = new.reshape(3,3)
        
        new_state = PuzzleState(new, 0, self)
        new_state.action_from_pred = direction
       
        return (new_state)
    
print('Artificial Intelligence')
print('MP1: A* for Sliding Puzzle')
print('SEMESTER: Fall 2019')
print('NAME: Jason Huggy')
print()

# load random start state onto frontier priority queue
frontier = queue.PriorityQueue()
a = np.loadtxt('mp1input.txt', dtype=np.int32)
start_state = PuzzleState(a,0,None)

frontier.put(start_state)

closed_set = set()

num_states = 0
while not frontier.empty():
    #  choose state at front of priority queue
    next_state = frontier.get()
    
    #  if goal then quit and return path
    if next_state.is_goal():
        next_state.show_path()
        break
    
    # Add state chosen for expansion to closed_set
    closed_set.add(next_state)
    num_states = num_states + 1
    
    # Expand state (up to 4 moves possible)
    possible_moves = ['up','down','left','right']
    for move in possible_moves:
        if next_state.can_move(move):
            neighbor = next_state.gen_next_state(move)
            if neighbor in closed_set:
                continue
            if neighbor not in frontier.queue:                           
                frontier.put(neighbor)
            # If it's already in the frontier, it's gauranteed to have lower cost, so no need to update

print('\nNumber of states visited =',num_states)