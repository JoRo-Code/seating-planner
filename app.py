import streamlit as st
import streamlit.web.bootstrap
import streamlit.components.v1 as components
import pandas as pd
import random
import math
import copy
from collections import defaultdict


#####################################
# 1. Define Tables, Seats & Naming  #
#####################################

# We assume three tables with different numbers of seats per side.
# For display purposes we will rename the tables as A, B, and C.
# (Internally we still use 0, 1, 2 as keys.)
TABLES = {
    0: 8,   # Table A: 8 seats per side (so 16 seats total)
    1: 10,  # Table B: 10 seats per side (20 seats total)
    2: 12   # Table C: 12 seats per side (24 seats total)
}

# Mapping from table id to table letter.
TABLE_LETTERS = {0: "A", 1: "B", 2: "C"}

def generate_seats(tables):
    """
    Returns a list of all seat positions.
    Each seat is represented as a tuple: (table, row, col)
    where table is an integer key (0,1,2), row is 0 or 1, and col in [0, length-1].
    """
    seats = []
    for t, length in tables.items():
        for row in [0, 1]:
            for col in range(length):
                seats.append((t, row, col))
    return seats

#####################################
# 2. Precompute Seat Neighbours     #
#####################################

def compute_seat_neighbors(tables):
    """
    For each seat, compute a list of neighbouring seats.
    Here, neighbours are defined as:
      - The immediate left and right in the same row.
      - In the opposite row: directly across plus the immediate left and right.
    Returns a dict mapping seat -> list of seats.
    """
    seat_neighbors = {}
    for t, length in tables.items():
        for row in [0, 1]:
            for col in range(length):
                s = (t, row, col)
                neighbours = []
                # Same row: left/right
                if col - 1 >= 0:
                    neighbours.append((t, row, col - 1))
                if col + 1 < length:
                    neighbours.append((t, row, col + 1))
                # Opposite row: directly across and diagonally left/right
                other = 1 - row
                neighbours.append((t, other, col))
                if col - 1 >= 0:
                    neighbours.append((t, other, col - 1))
                if col + 1 < length:
                    neighbours.append((t, other, col + 1))
                seat_neighbors[s] = neighbours
    return seat_neighbors

#####################################
# 3. Cost Function & Neighbour Info  #
#####################################

def compute_cost(assignments, seat_neighbors, tables, corner_penalty=3):
    """
    Computes a cost over all rounds based on:
      - Repeated neighbour meetings (if the same pair sits together in more than one round).
      - Sitting in a corner (col 0 or col L-1) more than once.
    'assignments' is a list (one per round) of dictionaries mapping seat -> person.
    """
    person_neighbors = defaultdict(list)
    person_corner_counts = defaultdict(int)
    
    for round_assign in assignments:
        for seat, person in round_assign.items():
            t, row, col = seat
            if col == 0 or col == tables[t] - 1:
                person_corner_counts[person] += 1
            for n_seat in seat_neighbors[seat]:
                neighbor_person = round_assign[n_seat]
                person_neighbors[person].append(neighbor_person)
    
    cost = 0
    for person, neighs in person_neighbors.items():
        cost += (len(neighs) - len(set(neighs)))
    for person, count in person_corner_counts.items():
        if count > 1:
            cost += (count - 1) * corner_penalty
    return cost

def get_neighbors_info(assignments, seat_neighbors, tables):
    """
    Returns a dictionary mapping each person to the set of unique neighbours
    (over all rounds) that they encountered.
    """
    person_neighbors = defaultdict(set)
    for round_assign in assignments:
        for seat, person in round_assign.items():
            for n_seat in seat_neighbors[seat]:
                neighbor_person = round_assign[n_seat]
                person_neighbors[person].add(neighbor_person)
    return person_neighbors

#####################################
# 4. Initialize Seating Assignments#
#####################################

def initialize_assignments(people, tables, fixed_positions, num_rounds=3):
    """
    Creates an initial seating assignment for each round.
    'fixed_positions' is a dict mapping person -> seat (tuple: (table, row, col))
    The remaining people are randomly assigned to the remaining seats.
    Returns a list of dictionaries, one per round.
    """
    seats = generate_seats(tables)
    free_people = set(people)
    for person in fixed_positions:
        free_people.discard(person)
    free_people = list(free_people)
    
    assignments = []  # One dictionary per round: seat -> person.
    for _ in range(num_rounds):
        round_assignment = {}
        # Place fixed positions first.
        for person, seat in fixed_positions.items():
            round_assignment[seat] = person
        # Randomly assign the remaining people.
        free_seats = [s for s in seats if s not in round_assignment]
        random.shuffle(free_seats)
        random.shuffle(free_people)
        for seat, person in zip(free_seats, free_people):
            round_assignment[seat] = person
        assignments.append(round_assignment)
    return assignments

#####################################
# 5. Optimize Assignments           #
#####################################

def optimize_assignments(assignments, seat_neighbors, tables, fixed_positions,
                         iterations=20000, initial_temp=10, cooling_rate=0.9995):
    """
    Uses simulated annealing to “optimize” the seating assignments.
    In each iteration, a random swap of two non-fixed seats in one round is attempted.
    Moves that lower the cost (or occasionally moves that worsen the cost) are accepted.
    Returns the best (lowest cost) assignments and the best cost.
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
        r = random.randint(0, num_rounds - 1)
        free_seats = free_seats_by_round[r]
        seat1, seat2 = random.sample(free_seats, 2)
        assignments[r][seat1], assignments[r][seat2] = assignments[r][seat2], assignments[r][seat1]
        new_cost = compute_cost(assignments, seat_neighbors, tables)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_assignments = copy.deepcopy(assignments)
        else:
            assignments[r][seat1], assignments[r][seat2] = assignments[r][seat2], assignments[r][seat1]
        temp *= cooling_rate
    return best_assignments, best_cost

#####################################
# 6. Create Seating Arrangement Tables
#####################################

def seating_dataframe_for_table(assignments, table_id, table_letter):
    """
    Given the list of assignments (one per round), extract the seats for the given table.
    The seats are sorted (first by row then by column) and then relabeled as table_letter1, table_letter2, etc.
    Returns a DataFrame with index as seat labels and one column per arrangement.
    """
    # Use the first round's assignment to determine all seats for this table.
    seats = [seat for seat in assignments[0].keys() if seat[0] == table_id]
    seats.sort(key=lambda s: (s[1], s[2]))  # sort by row then col
    labels = [f"{table_letter}{i+1}" for i in range(len(seats))]
    data = {}
    for round_idx, assignment in enumerate(assignments):
        # For each round, get the occupant for each seat in sorted order.
        persons = [assignment[seat] for seat in seats]
        data[f"Arrangement {round_idx+1}"] = persons
    df = pd.DataFrame(data, index=labels)
    df.index.name = f"Table {table_letter} Seat"
    return df

#####################################
# 7. Run Optimization & Build Data  #
#####################################

def run_optimization_and_build_data(iterations, initial_temp, cooling_rate):
    """
    Runs the optimization process and returns:
      - best_assignments: a list (per round) of seat -> person assignments,
      - best_cost: the final cost,
      - neighbors_info: a dict mapping each person to their overall set of neighbours.
    """
    # Total number of seats = sum over tables (each table has 2 sides).
    total_seats = sum(2 * length for length in TABLES.values())
    # Use short names: P1, P2, etc.
    people = [f"P{i+1}" for i in range(total_seats)]
    # Example fixed positions (if desired).
    fixed_positions = {
        "P1": (0, 0, 0),   # Always at Table A, side 0, seat 0.
        "P2": (1, 1, 9),   # Always at Table B, side 1, seat 9.
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

#####################################
# 8. Main App                        #
#####################################

def main():
    st.title("Seating Arrangement Summary")
    st.markdown(
        """
        This app “optimizes” seating arrangements over 3 rounds so that people sit with as many different people as possible.
        It then displays a summary for each table (labeled A, B, and C) with the seating for each arrangement.
        Finally, it shows the overall neighbour list (the set of people who sat next to each person, across all rounds).
        """
    )
    
    # Sidebar options for optimization parameters.
    st.sidebar.header("Optimization Parameters")
    iterations = st.sidebar.number_input("Iterations", value=20000, step=1000, min_value=1000)
    initial_temp = st.sidebar.number_input("Initial Temperature", value=10.0, step=1.0)
    cooling_rate = st.sidebar.slider("Cooling Rate", min_value=0.990, max_value=0.9999, value=0.9995)
    run_button = st.sidebar.button("Run Optimization")
    
    # Run optimization and store results in session_state so that re-runs are faster.
    if run_button or "best_assignments" not in st.session_state:
        with st.spinner("Running optimization..."):
            best_assignments, best_cost, neighbors_info = run_optimization_and_build_data(iterations, initial_temp, cooling_rate)
            st.session_state.best_assignments = best_assignments
            st.session_state.best_cost = best_cost
            st.session_state.neighbors_info = neighbors_info
    
    st.success(f"Optimization complete. Best cost: {st.session_state.best_cost}")
    
    # Display seating arrangements for each table.
    st.header("Seating Arrangements by Table")
    for table_id in sorted(TABLES.keys()):
        table_letter = TABLE_LETTERS[table_id]
        df = seating_dataframe_for_table(st.session_state.best_assignments, table_id, table_letter)
        st.subheader(f"Table {table_letter} (Total seats: {len(df)})")
        st.dataframe(df, height=300)
    
    # Display overall neighbour summary.
    st.header("Overall Neighbour Summary")
    # Convert the neighbors_info dict to a DataFrame.
    data = []
    for person, neighbours in st.session_state.neighbors_info.items():
        data.append((person, ", ".join(sorted(neighbours))))
    nbr_df = pd.DataFrame(data, columns=["Person", "Neighbours"])
    st.dataframe(nbr_df, height=400)

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
