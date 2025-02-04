import pandas as pd
import random
import math
import copy
from collections import defaultdict

#############################
# 1. Define Tables & Seats  #
#############################

# We assume three tables with different lengths.
TABLES = {
    0: 8,   # Table 0 has 8 seats per side
    1: 10,  # Table 1 has 10 seats per side
    2: 12   # Table 2 has 12 seats per side
}

def generate_seats(tables):
    """
    Create a list of all seat positions.
    Each seat is represented as a tuple: (table_id, row, col)
    where row is 0 or 1 (two sides) and col runs from 0 to table_length-1.
    """
    seats = []
    for t, length in tables.items():
        for row in [0, 1]:
            for col in range(length):
                seats.append((t, row, col))
    return seats

#####################################
# 2. Precompute Each Seat’s Neighbors #
#####################################

def compute_seat_neighbors(tables):
    """
    For each seat, we return a list of its neighbour seats.
    We define neighbours as:
      - In the same row: the seat immediately to the left and to the right.
      - In the opposite row: the seat directly across, and the seats diagonally left/right.
    """
    seat_neighbors = {}
    for t, length in tables.items():
        for row in [0, 1]:
            for col in range(length):
                s = (t, row, col)
                neighbours = []
                # Same row: left/right (if within bounds)
                if col - 1 >= 0:
                    neighbours.append((t, row, col-1))
                if col + 1 < length:
                    neighbours.append((t, row, col+1))
                # Opposite row: directly across and diagonal left/right.
                other = 1 - row
                neighbours.append((t, other, col))
                if col - 1 >= 0:
                    neighbours.append((t, other, col-1))
                if col + 1 < length:
                    neighbours.append((t, other, col+1))
                seat_neighbors[s] = neighbours
    return seat_neighbors

#########################################
# 3. Build the Cost Function & Neighbors #
#########################################

def compute_cost(assignments, seat_neighbors, tables, corner_penalty=3):
    """
    For every person over all rounds, compute a cost that:
      - Penalizes a person for having the same neighbour more than once.
      - Penalizes if the person sits in a corner more than once.
    
    assignments is a list of dicts (one per round) mapping seat->person.
    """
    # For each person, collect the list of neighbours (neighbors may be repeated across rounds)
    person_neighbors = defaultdict(list)
    # Count how many times each person sits in a "corner" (col 0 or col L-1 for that table)
    person_corner_counts = defaultdict(int)
    
    for round_assign in assignments:
        for seat, person in round_assign.items():
            t, row, col = seat
            # Check for corner: if col==0 or col==table_length-1.
            if col == 0 or col == tables[t]-1:
                person_corner_counts[person] += 1
            for n_seat in seat_neighbors[seat]:
                # We assume every seat is occupied.
                neighbor_person = round_assign[n_seat]
                person_neighbors[person].append(neighbor_person)
    
    cost = 0
    # For each person, add the “repetition cost” (total neighbour count minus the number of unique neighbours)
    for person, neighs in person_neighbors.items():
        cost += (len(neighs) - len(set(neighs)))
    
    # Add a penalty if a person sits in a corner more than once.
    for person, count in person_corner_counts.items():
        if count > 1:
            cost += (count - 1) * corner_penalty

    return cost

def get_neighbors_info(assignments, seat_neighbors, tables):
    """
    Return a dictionary mapping each person to the set of unique neighbours they met
    over all seatings.
    """
    person_neighbors = defaultdict(set)
    for round_assign in assignments:
        for seat, person in round_assign.items():
            for n_seat in seat_neighbors[seat]:
                neighbor_person = round_assign[n_seat]
                person_neighbors[person].add(neighbor_person)
    return person_neighbors

##############################################
# 4. Initialize Seating Assignments          #
##############################################

def initialize_assignments(people, tables, fixed_positions, num_rounds=3):
    """
    Create an initial assignment for each seating round.
    - fixed_positions is a dict mapping person -> seat (a tuple (table, row, col))
      meaning that this person must always sit in that seat.
    - The rest of the persons are randomly assigned to the remaining seats in each round.
    """
    seats = generate_seats(tables)
    
    # Make a copy of the free people list (those not fixed)
    free_people = set(people)
    for person in fixed_positions:
        free_people.discard(person)
    free_people = list(free_people)
    
    assignments = []  # a list of dictionaries, one per round, mapping seat->person
    for _ in range(num_rounds):
        round_assignment = {}
        # First, fill in the fixed seats.
        for person, seat in fixed_positions.items():
            round_assignment[seat] = person
        # Next, assign the free people to the remaining seats.
        free_seats = [s for s in seats if s not in round_assignment]
        random.shuffle(free_seats)
        random.shuffle(free_people)
        for seat, person in zip(free_seats, free_people):
            round_assignment[seat] = person
        assignments.append(round_assignment)
    return assignments

##################################################
# 5. Optimize the Arrangements (Simulated Annealing) #
##################################################

def optimize_assignments(assignments, seat_neighbors, tables, fixed_positions,
                         iterations=20000, initial_temp=10, cooling_rate=0.9995):
    """
    Our optimization loop randomly picks a seating round and swaps two persons in seats
    that are not fixed. Moves that lower the cost (or occasionally moves that increase cost
    a little) are accepted.
    """
    current_cost = compute_cost(assignments, seat_neighbors, tables)
    best_cost = current_cost
    best_assignments = copy.deepcopy(assignments)
    num_rounds = len(assignments)
    
    # Pre-calculate free seats for each round (those that are not fixed)
    seats = generate_seats(tables)
    free_seats_by_round = []
    for _ in range(num_rounds):
        fixed = set(fixed_positions.values())
        free_seats_by_round.append([s for s in seats if s not in fixed])
    
    temp = initial_temp
    for it in range(iterations):
        # Choose a random round and two free seats in that round.
        r = random.randint(0, num_rounds - 1)
        free_seats = free_seats_by_round[r]
        seat1, seat2 = random.sample(free_seats, 2)
        
        # Swap the persons in seat1 and seat2.
        assignments[r][seat1], assignments[r][seat2] = assignments[r][seat2], assignments[r][seat1]
        
        new_cost = compute_cost(assignments, seat_neighbors, tables)
        delta = new_cost - current_cost
        # Accept if the move lowers the cost, or sometimes if it doesn’t.
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_assignments = copy.deepcopy(assignments)
        else:
            # Revert the swap.
            assignments[r][seat1], assignments[r][seat2] = assignments[r][seat2], assignments[r][seat1]
        
        temp *= cooling_rate
        if it % 1000 == 0:
            print(f"Iteration {it}: current_cost = {current_cost}, best_cost = {best_cost}")
    
    return best_assignments, best_cost

###############################################
# 6. Visualize the Seating & Neighbour Results #
###############################################

def visualize_assignments(assignments, tables):
    """
    For each round and each table, create a pandas DataFrame.
    The DataFrame has two rows (one per side) and a column for each seat position.
    """
    visuals = {}
    num_rounds = len(assignments)
    for r in range(num_rounds):
        visuals[r] = {}
        for t, length in tables.items():
            df = pd.DataFrame(index=[0, 1], columns=list(range(length)))
            for row in [0, 1]:
                for col in range(length):
                    seat = (t, row, col)
                    df.at[row, col] = assignments[r].get(seat, "")
            visuals[r][t] = df
    return visuals

def visualize_neighbors(neighbors_info):
    """
    Convert the neighbour info dictionary into a DataFrame.
    """
    data = []
    for person, neighs in neighbors_info.items():
        data.append((person, ", ".join(sorted(neighs))))
    df = pd.DataFrame(data, columns=["Person", "Neighbours"])
    return df

###############################
# 7. Main Program Entry Point #
###############################

def main():
    # For this example we assume the number of people equals the total number of seats.
    total_seats = sum(2 * length for length in TABLES.values())
    people = [f"P{i+1}" for i in range(total_seats)]
    
    # Optionally fix some people to specific seats.
    # For example, here Person_1 always sits at Table 0, row 0, col 0
    # and Person_2 always sits at Table 1, row 1, col 9.
    fixed_positions = {
        "P1": (0, 0, 0),
        "P2": (1, 1, 9),
        # You can add more fixed assignments as needed.
    }
    
    # Precompute the neighbour mapping.
    seat_neighbors = compute_seat_neighbors(TABLES)
    
    # Generate an initial seating plan (3 rounds).
    assignments = initialize_assignments(people, TABLES, fixed_positions, num_rounds=3)
    
    # Optimize the seating arrangements.
    optimized_assignments, best_cost = optimize_assignments(assignments, seat_neighbors, TABLES,
                                                            fixed_positions,
                                                            iterations=20000,
                                                            initial_temp=10,
                                                            cooling_rate=0.9995)
    print("\nOptimization complete. Best cost:", best_cost)
    
    # Visualize the seating arrangements.
    visuals = visualize_assignments(optimized_assignments, TABLES)
    for r in visuals:
        print(f"\n=== Seating Round {r+1} ===")
        for t, df in visuals[r].items():
            print(f"\nTable {t} (Length {TABLES[t]}):")
            print(df)
    
    # Compute and show the neighbour info for each person.
    neighbors_info = get_neighbors_info(optimized_assignments, seat_neighbors, TABLES)
    neighbors_df = visualize_neighbors(neighbors_info)
    print("\n=== Neighbours for Each Person Over All Seatings ===")
    print(neighbors_df.to_string(index=False))

if __name__ == "__main__":
    main()
