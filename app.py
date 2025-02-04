import streamlit as st
import streamlit.web.bootstrap

import pandas as pd
import random
import math
import copy
from collections import defaultdict

#############################
# 1. Define Tables & Seats  #
#############################

# Three tables with different lengths.
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
# 2. Precompute Each Seat’s Neighbours #
#####################################

def compute_seat_neighbors(tables):
    """
    For each seat, return a list of its neighbour seats.
    Neighbours are defined as:
      - In the same row: immediate left/right.
      - In the opposite row: directly across plus diagonals.
    """
    seat_neighbors = {}
    for t, length in tables.items():
        for row in [0, 1]:
            for col in range(length):
                s = (t, row, col)
                neighbours = []
                # Same row: left/right.
                if col - 1 >= 0:
                    neighbours.append((t, row, col - 1))
                if col + 1 < length:
                    neighbours.append((t, row, col + 1))
                # Opposite row: directly across and diagonal left/right.
                other = 1 - row
                neighbours.append((t, other, col))
                if col - 1 >= 0:
                    neighbours.append((t, other, col - 1))
                if col + 1 < length:
                    neighbours.append((t, other, col + 1))
                seat_neighbors[s] = neighbours
    return seat_neighbors

#########################################
# 3. Build the Cost Function & Neighbour Collection #
#########################################

def compute_cost(assignments, seat_neighbors, tables, corner_penalty=3):
    """
    For every person over all rounds, compute a cost that:
      - Penalizes a person for re-encountering the same neighbour.
      - Penalizes if the person sits in a corner (col 0 or col L-1) more than once.
    
    assignments is a list of dicts (one per round) mapping seat -> person.
    """
    person_neighbors = defaultdict(list)
    person_corner_counts = defaultdict(int)
    
    for round_assign in assignments:
        for seat, person in round_assign.items():
            t, row, col = seat
            # Check for a corner seat.
            if col == 0 or col == tables[t] - 1:
                person_corner_counts[person] += 1
            for n_seat in seat_neighbors[seat]:
                neighbor_person = round_assign[n_seat]
                person_neighbors[person].append(neighbor_person)
    
    cost = 0
    # Cost for repeated neighbours.
    for person, neighs in person_neighbors.items():
        cost += (len(neighs) - len(set(neighs)))
    
    # Add penalty for sitting in a corner more than once.
    for person, count in person_corner_counts.items():
        if count > 1:
            cost += (count - 1) * corner_penalty

    return cost

def get_neighbors_info(assignments, seat_neighbors, tables):
    """
    Return a dictionary mapping each person to the set of unique neighbours they met.
    """
    person_neighbors = defaultdict(set)
    for round_assign in assignments:
        for seat, person in round_assign.items():
            for n_seat in seat_neighbors[seat]:
                neighbor_person = round_assign[n_seat]
                person_neighbors[person].add(neighbor_person)
    return person_neighbors

#########################################
# 4. Initialize Seating Assignments    #
#########################################

def initialize_assignments(people, tables, fixed_positions, num_rounds=3):
    """
    Create an initial assignment for each seating round.
    - fixed_positions is a dict mapping person -> seat (tuple: (table, row, col))
      meaning that this person must always sit in that seat.
    - The rest of the persons are randomly assigned to remaining seats.
    """
    seats = generate_seats(tables)
    free_people = set(people)
    for person in fixed_positions:
        free_people.discard(person)
    free_people = list(free_people)
    
    assignments = []  # List of dicts: one per round mapping seat -> person.
    for _ in range(num_rounds):
        round_assignment = {}
        # Place fixed positions.
        for person, seat in fixed_positions.items():
            round_assignment[seat] = person
        # Assign free people to remaining seats.
        free_seats = [s for s in seats if s not in round_assignment]
        random.shuffle(free_seats)
        random.shuffle(free_people)
        for seat, person in zip(free_seats, free_people):
            round_assignment[seat] = person
        assignments.append(round_assignment)
    return assignments

##############################################
# 5. Optimize the Arrangements (Simulated Annealing) #
##############################################

def optimize_assignments(assignments, seat_neighbors, tables, fixed_positions,
                         iterations=20000, initial_temp=10, cooling_rate=0.9995):
    """
    Optimization loop: randomly pick a seating round and swap two free (non-fixed) seats.
    Moves that lower the cost (or slightly worse moves with some probability) are accepted.
    """
    current_cost = compute_cost(assignments, seat_neighbors, tables)
    best_cost = current_cost
    best_assignments = copy.deepcopy(assignments)
    num_rounds = len(assignments)
    
    # Pre-calculate free seats per round.
    seats = generate_seats(tables)
    free_seats_by_round = []
    for _ in range(num_rounds):
        fixed = set(fixed_positions.values())
        free_seats_by_round.append([s for s in seats if s not in fixed])
    
    temp = initial_temp
    for _ in range(iterations):
        # Pick a random round and two free seats.
        r = random.randint(0, num_rounds - 1)
        free_seats = free_seats_by_round[r]
        seat1, seat2 = random.sample(free_seats, 2)
        
        # Swap the persons in seat1 and seat2.
        assignments[r][seat1], assignments[r][seat2] = assignments[r][seat2], assignments[r][seat1]
        
        new_cost = compute_cost(assignments, seat_neighbors, tables)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_assignments = copy.deepcopy(assignments)
        else:
            # Revert swap.
            assignments[r][seat1], assignments[r][seat2] = assignments[r][seat2], assignments[r][seat1]
        
        temp *= cooling_rate
    
    return best_assignments, best_cost

###############################################
# 6. Visualization Functions                 #
###############################################

def visualize_assignments(assignments, tables):
    """
    For each round and each table, create a pandas DataFrame.
    The DataFrame has two rows (one per side) and one column per seat.
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

###############################################
# 7. Run Optimization (Used by the Streamlit App) #
###############################################

def run_optimization(iterations, initial_temp, cooling_rate):
    # Assume the number of people equals the total number of seats.
    total_seats = sum(2 * length for length in TABLES.values())
    people = [f"P{i+1}" for i in range(total_seats)]
    
    # Example fixed positions:
    fixed_positions = {
        "P1": (0, 0, 0),   # Always at Table 0, side 0, seat 0.
        "P2": (1, 1, 9),   # Always at Table 1, side 1, seat 9.
    }
    
    seat_neighbors = compute_seat_neighbors(TABLES)
    assignments = initialize_assignments(people, TABLES, fixed_positions, num_rounds=3)
    best_assignments, best_cost = optimize_assignments(assignments, seat_neighbors, TABLES,
                                                       fixed_positions,
                                                       iterations=iterations,
                                                       initial_temp=initial_temp,
                                                       cooling_rate=cooling_rate)
    neighbors_info = get_neighbors_info(best_assignments, seat_neighbors, TABLES)
    return best_assignments, best_cost, neighbors_info

###############################################
# 8. Streamlit App Main Function             #
###############################################

def main():
    st.title("Seating Arrangement Optimization")
    st.markdown(
        """
        This app generates seating arrangements over several rounds so that all people get to sit with as many different people as possible.
        Some positions can be fixed (e.g., Person_1 and Person_2), and the optimizer will try to avoid having people repeatedly sitting at table corners.
        """
    )
    
    # Sidebar options to adjust parameters.
    st.sidebar.header("Optimization Parameters")
    iterations = st.sidebar.number_input("Iterations", value=20000, step=1000, min_value=1000)
    initial_temp = st.sidebar.number_input("Initial Temperature", value=10.0, step=1.0)
    cooling_rate = st.sidebar.slider("Cooling Rate", min_value=0.990, max_value=0.9999, value=0.9995)
    
    run_button = st.sidebar.button("Run Optimization")
    
    if run_button:
        st.info("Running optimization, please wait...")
        best_assignments, best_cost, neighbors_info = run_optimization(iterations, initial_temp, cooling_rate)
        st.success(f"Optimization complete. Best cost: {best_cost}")
        
        # Visualize seating arrangements.
        visuals = visualize_assignments(best_assignments, TABLES)
        for r in visuals:
            st.subheader(f"Seating Round {r+1}")
            for t, df in visuals[r].items():
                st.markdown(f"**Table {t} (Length {TABLES[t]})**")
                st.dataframe(df)
        
        # Visualize neighbours summary.
        neighbors_df = visualize_neighbors(neighbors_info)
        st.subheader("Neighbours for Each Person Over All Seatings")
        st.dataframe(neighbors_df)

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
