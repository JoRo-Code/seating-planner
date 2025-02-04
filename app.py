import streamlit as st
import streamlit.web.bootstrap
import streamlit.components.v1 as components
import pandas as pd
import random
import math
import copy
from collections import defaultdict

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import random
import math
import copy
from collections import defaultdict

#####################################
# 1. Table Definitions & Utilities  #
#####################################

# We assume three tables with different numbers of seats per side.
# For display, we label the tables as A, B, and C.
TABLES = {
    0: 8,    # Table A: 8 seats per side → 16 seats total
    1: 10,   # Table B: 10 seats per side → 20 seats total
    2: 12    # Table C: 12 seats per side → 24 seats total
}
TABLE_LETTERS = {0: "A", 1: "B", 2: "C"}

def generate_seats(tables):
    """Return a list of all seats as tuples (table, row, col)."""
    seats = []
    for t, length in tables.items():
        for row in [0, 1]:
            for col in range(length):
                seats.append((t, row, col))
    return seats

def compute_seat_neighbors(tables):
    """
    For each seat, compute a list of neighbouring seats.
    Neighbours are:
      - Immediate left/right in the same row.
      - In the opposite row: directly across plus the adjacent left/right.
    Returns a dict mapping seat -> list of seats.
    """
    seat_neighbors = {}
    for t, length in tables.items():
        for row in [0, 1]:
            for col in range(length):
                s = (t, row, col)
                neighbours = []
                if col - 1 >= 0:
                    neighbours.append((t, row, col - 1))
                if col + 1 < length:
                    neighbours.append((t, row, col + 1))
                other = 1 - row
                neighbours.append((t, other, col))
                if col - 1 >= 0:
                    neighbours.append((t, other, col - 1))
                if col + 1 < length:
                    neighbours.append((t, other, col + 1))
                seat_neighbors[s] = neighbours
    return seat_neighbors

#####################################
# 2. Optimization Functions          #
#####################################

def compute_cost(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                 neighbor_weight=1.0, corner_weight=3.0, gender_weight=5.0, fixed_weight=2.0):
    """
    Computes total cost over all rounds as a weighted sum of:
      - Neighbor repetition: (total neighbors - unique neighbors) per person.
        For fixed-seat persons, the penalty is multiplied by fixed_weight.
      - Corner cost: if a person sits in a corner (col 0 or col L-1) more than once.
      - Gender cost: for each adjacent pair in a row that has the same gender.
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
    neighbor_cost = 0
    for person, neighs in person_neighbors.items():
        if person_genders.get(person, "X") == "X":
            continue
        multiplier = fixed_weight if person in fixed_positions else 1.0
        neighbor_cost += multiplier * (len(neighs) - len(set(neighs)))
    corner_cost = 0
    for person, count in person_corner_counts.items():
        if person_genders.get(person, "X") == "X":
            continue
        if count > 1:
            corner_cost += (count - 1)
    gender_cost = 0
    for round_assign in assignments:
        for t, length in tables.items():
            for row in [0, 1]:
                seats_in_row = [(t, row, col) for col in range(length)]
                for i in range(len(seats_in_row) - 1):
                    s1, s2 = seats_in_row[i], seats_in_row[i+1]
                    p1 = round_assign[s1]
                    p2 = round_assign[s2]
                    g1 = person_genders.get(p1, "X")
                    g2 = person_genders.get(p2, "X")
                    if g1 == "X" or g2 == "X":
                        continue
                    if g1 == g2:
                        gender_cost += 1
    total_cost = (neighbor_weight * neighbor_cost +
                  corner_weight * corner_cost +
                  gender_weight * gender_cost)
    return total_cost

def get_neighbors_info(assignments, seat_neighbors, tables):
    """Returns a dict mapping each person to the set of unique neighbours over all rounds."""
    person_neighbors = defaultdict(set)
    for round_assign in assignments:
        for seat, person in round_assign.items():
            for n_seat in seat_neighbors[seat]:
                neighbor_person = round_assign[n_seat]
                person_neighbors[person].add(neighbor_person)
    return person_neighbors

def initialize_assignments(people, tables, fixed_positions, num_rounds=3):
    """
    Creates an initial seating assignment (one per round).
    Fixed seats are pre-assigned; remaining seats are randomly filled.
    Returns a list of dictionaries mapping seat -> person.
    """
    seats = generate_seats(tables)
    free_people = set(people)
    for person in fixed_positions:
        free_people.discard(person)
    free_people = list(free_people)
    assignments = []
    for _ in range(num_rounds):
        round_assignment = {}
        for person, seat in fixed_positions.items():
            round_assignment[seat] = person
        free_seats = [s for s in seats if s not in round_assignment]
        random.shuffle(free_seats)
        random.shuffle(free_people)
        for seat, person in zip(free_seats, free_people):
            round_assignment[seat] = person
        assignments.append(round_assignment)
    return assignments

def optimize_assignments(assignments, seat_neighbors, tables, fixed_positions, person_genders,
                         iterations=20000, initial_temp=10, cooling_rate=0.9995,
                         neighbor_weight=1.0, corner_weight=3.0, gender_weight=5.0, fixed_weight=2.0):
    """
    Uses simulated annealing to optimize seating assignments.
    In each iteration, two non-fixed seats in a random round are swapped.
    Returns the best assignments and the corresponding cost.
    """
    current_cost = compute_cost(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                                neighbor_weight, corner_weight, gender_weight, fixed_weight)
    best_cost = current_cost
    best_assignments = copy.deepcopy(assignments)
    num_rounds = len(assignments)
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
        new_cost = compute_cost(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                                neighbor_weight, corner_weight, gender_weight, fixed_weight)
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

def seating_dataframe_for_table(assignments, table_id, table_letter):
    """
    For the given table, sorts its seats (by row then column) and builds a DataFrame.
    Each row is labeled (e.g. A1, A2, …) and each column corresponds to an arrangement.
    """
    seats = [seat for seat in assignments[0].keys() if seat[0] == table_id]
    seats.sort(key=lambda s: (s[1], s[2]))
    labels = [f"{table_letter}{i+1}" for i in range(len(seats))]
    data = {}
    for round_idx, assignment in enumerate(assignments):
        persons = [assignment[seat] for seat in seats]
        data[f"Arrangement {round_idx+1}"] = persons
    df = pd.DataFrame(data, index=labels)
    df.index.name = f"Table {table_letter} Seat"
    return df

def run_optimization_and_build_data(iterations, initial_temp, cooling_rate,
                                      neighbor_weight, corner_weight, gender_weight, fixed_weight,
                                      people, person_genders, fixed_positions):
    """
    Runs optimization and returns the best assignments, best cost, and neighbor info.
    If there are fewer names than seats, fills remaining seats with "Empty" (neutral gender "X").
    """
    total_seats = sum(2 * length for length in TABLES.values())
    if len(people) < total_seats:
        num_missing = total_seats - len(people)
        for i in range(num_missing):
            empty_name = f"Empty{i+1}"
            people.append(empty_name)
            person_genders[empty_name] = "X"
    elif len(people) > total_seats:
        people = people[:total_seats]
    seat_neighbors = compute_seat_neighbors(TABLES)
    assignments = initialize_assignments(people, TABLES, fixed_positions, num_rounds=3)
    best_assignments, best_cost = optimize_assignments(assignments, seat_neighbors, TABLES,
                                                       fixed_positions, person_genders,
                                                       iterations, initial_temp, cooling_rate,
                                                       neighbor_weight, corner_weight, gender_weight, fixed_weight)
    neighbors_info = get_neighbors_info(best_assignments, seat_neighbors, TABLES)
    return best_assignments, best_cost, neighbors_info

#####################################
# 3. Fixed Seat Input Parser         #
#####################################

def parse_fixed_seats(text):
    """
    Parses fixed seat assignments from the given text.
    Each line should be in the format: Name: TableLetter,Row,Col (e.g., John: A,0,3)
    Returns a dict mapping Name -> (table, row, col)
    """
    fixed = {}
    lines = text.strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        try:
            name_part, seat_part = line.split(":", 1)
            name = name_part.strip()
            parts = seat_part.strip().split(",")
            if len(parts) != 3:
                continue
            table_letter = parts[0].strip().upper()
            row = int(parts[1].strip())
            col = int(parts[2].strip())
            table_id = None
            for tid, letter in TABLE_LETTERS.items():
                if letter == table_letter:
                    table_id = tid
                    break
            if table_id is None:
                continue
            fixed[name] = (table_id, row, col)
        except Exception as e:
            continue
    return fixed

#####################################
# 4. Table Layout Visualization       #
#####################################

def generate_table_html(table_id, table_letter):
    """
    Generates HTML representing the layout of one table.
    Seats are labeled sequentially (top row first, then bottom row, e.g. A1, A2, …).
    Corner seats (leftmost and rightmost in each row) are highlighted in light red.
    """
    num_cols = TABLES[table_id]
    top_row = [f"{table_letter}{i+1}" for i in range(num_cols)]
    bottom_row = [f"{table_letter}{i+1+num_cols}" for i in range(num_cols)]
    cell_style = (
        "width:40px; height:40px; border:1px solid #000; "
        "display:flex; align-items:center; justify-content:center; margin:2px; "
        "font-size:12px; font-weight:bold;"
    )
    corner_bg = "#ffcccc"  # light red for corners
    normal_bg = "#ffffff"
    def build_row_html(row_labels):
        row_html = "<div style='display:flex; justify-content:center;'>"
        for idx, label in enumerate(row_labels):
            bg = corner_bg if idx == 0 or idx == num_cols - 1 else normal_bg
            cell_html = f"<div style='{cell_style} background-color:{bg};'>{label}</div>"
            row_html += cell_html
        row_html += "</div>"
        return row_html
    top_html = build_row_html(top_row)
    bottom_html = build_row_html(bottom_row)
    full_html = f"""
    <html>
      <head>
        <meta charset="UTF-8">
        <style>
          body {{ font-family: sans-serif; margin:10px; padding:0; }}
        </style>
      </head>
      <body>
        <h3 style="text-align:center;">Table {table_letter}</h3>
        {top_html}
        {bottom_html}
      </body>
    </html>
    """
    return full_html

def display_table_layouts():
    st.markdown("## Table Layouts")
    st.markdown(
        """
        The following displays the seat numbers for each table.
        For example, Table A’s seats are labeled A1, A2, … etc.
        Corner seats are highlighted in light red.
        """
    )
    for table_id in sorted(TABLES.keys()):
        table_letter = TABLE_LETTERS[table_id]
        html = generate_table_html(table_id, table_letter)
        components.html(html, height=180, scrolling=False)

#####################################
# 5. Main App                        #
#####################################

def main():
    st.title("Seating Arrangement Optimizer & Table Layout Visualizer")
    
    # Let the user choose the view.
    view = st.sidebar.radio("Select View", ["Seating Arrangements", "Table Layouts"])
    
    if view == "Seating Arrangements":
        st.header("Seating Arrangement Optimization with Fixed Seat Diversity")
        st.markdown(
            """
            This section optimizes seating arrangements over 3 rounds with the following conditions:
              - People should sit with as many different people as possible.
              - Minimize repeated corner seating.
              - Alternate genders in each table row (if possible).
              - For fixed seats (pre-assigned), encourage more diversity among neighbors.
              
            Provide lists of male and female names (one per line) and specify fixed seat assignments.
            If there are fewer names than available seats, the remaining seats will be filled with "Empty" placeholders.
            """
        )
        
        st.sidebar.header("Name Lists")
        default_male = "John\nMike\nDavid\nSteve\nRobert\nJames\nWilliam\nRichard\nJoseph\nThomas"
        default_female = "Mary\nLinda\nSusan\nKaren\nPatricia\nBarbara\nNancy\nLisa\nBetty\nMargaret"
        male_text = st.sidebar.text_area("Male Names (one per line)", value=default_male, height=150)
        female_text = st.sidebar.text_area("Female Names (one per line)", value=default_female, height=150)
        male_names = [name.strip() for name in male_text.splitlines() if name.strip()]
        female_names = [name.strip() for name in female_text.splitlines() if name.strip()]
        people = male_names + female_names
        person_genders = {}
        for name in male_names:
            person_genders[name] = "M"
        for name in female_names:
            person_genders[name] = "F"
        
        st.sidebar.header("Fixed Seat Assignments")
        fixed_text = st.sidebar.text_area("Enter fixed assignments (e.g. `John: A,0,3`)", value="John: A,0,0\nMary: B,1,5", height=100)
        fixed_positions = parse_fixed_seats(fixed_text)
        
        st.sidebar.header("Optimization Parameters")
        iterations = st.sidebar.number_input("Iterations", value=20000, step=1000, min_value=1000)
        initial_temp = st.sidebar.number_input("Initial Temperature", value=10.0, step=1.0)
        cooling_rate = st.sidebar.slider("Cooling Rate", min_value=0.990, max_value=0.9999, value=0.9995)
        
        st.sidebar.header("Condition Weights")
        neighbor_weight = st.sidebar.number_input("Neighbor Weight", value=1.0, step=0.1, format="%.1f")
        corner_weight = st.sidebar.number_input("Corner Weight", value=3.0, step=0.1, format="%.1f")
        gender_weight = st.sidebar.number_input("Gender Weight", value=5.0, step=0.1, format="%.1f")
        fixed_weight = st.sidebar.number_input("Fixed Seat Diversity Weight", value=2.0, step=0.1, format="%.1f")
        
        run_button = st.sidebar.button("Run Optimization")
        
        if run_button or "best_assignments" not in st.session_state:
            with st.spinner("Running optimization..."):
                best_assignments, best_cost, neighbors_info = run_optimization_and_build_data(
                    iterations, initial_temp, cooling_rate,
                    neighbor_weight, corner_weight, gender_weight, fixed_weight,
                    people, person_genders, fixed_positions
                )
                st.session_state.best_assignments = best_assignments
                st.session_state.best_cost = best_cost
                st.session_state.neighbors_info = neighbors_info
                st.session_state.person_genders = person_genders
        
        st.success(f"Optimization complete. Best cost: {st.session_state.best_cost}")
        st.header("Seating Arrangements by Table")
        for table_id in sorted(TABLES.keys()):
            table_letter = TABLE_LETTERS[table_id]
            df = seating_dataframe_for_table(st.session_state.best_assignments, table_id, table_letter)
            st.subheader(f"Table {table_letter} (Total seats: {len(df)})")
            st.dataframe(df, height=300)
        
        st.header("Overall Neighbour Summary")
        data = []
        for person, neighbours in st.session_state.neighbors_info.items():
            data.append((person, st.session_state.person_genders.get(person, "X"), ", ".join(sorted(neighbours))))
        nbr_df = pd.DataFrame(data, columns=["Person", "Gender", "Neighbours"])
        st.dataframe(nbr_df, height=400)
    
    elif view == "Table Layouts":
        display_table_layouts()

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
