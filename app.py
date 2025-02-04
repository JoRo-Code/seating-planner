import streamlit as st
import streamlit.web.bootstrap
import streamlit.components.v1 as components
import pandas as pd
import random
import math
import copy
from collections import defaultdict
import json

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

# Default table definitions (if the user does not change them)
DEFAULT_TABLE_DEF = "A: 3\nB: 3\nC: 4"

def parse_table_definitions(text):
    """
    Parses table definitions from text.
    Each line should be in the format: Letter: Number
    For example: "A: 8" means Table A has 8 seats per side.
    Returns two dictionaries:
      - tables: mapping integer table id (0,1,2,...) → seats per side.
      - table_letters: mapping table id → table letter.
    """
    lines = text.strip().splitlines()
    tables_list = []
    for line in lines:
        if not line.strip():
            continue
        try:
            letter, seats_str = line.split(":")
            letter = letter.strip().upper()
            seats_per_side = int(seats_str.strip())
            tables_list.append((letter, seats_per_side))
        except Exception:
            continue
    tables_list.sort(key=lambda x: x[0])
    tables = {}
    table_letters = {}
    for idx, (letter, seats_per_side) in enumerate(tables_list):
        tables[idx] = seats_per_side
        table_letters[idx] = letter
    return tables, table_letters

#####################################
# 2. Seat and Neighbor Utilities    #
#####################################

def generate_seats(tables):
    """
    Returns a list of all seat positions.
    Each seat is a tuple: (table, row, col)
    where table is an integer key, row is 0 or 1, and col in [0, seats per side - 1].
    """
    seats = []
    for t, seats_per_side in tables.items():
        for row in [0, 1]:
            for col in range(seats_per_side):
                seats.append((t, row, col))
    return seats

def compute_seat_neighbors(tables):
    """
    For each seat, computes a list of neighboring seats.
    Neighbors are:
      - Immediate left/right in the same row.
      - In the opposite row: directly across plus the adjacent left/right.
    Returns a dictionary mapping seat -> list of seats.
    """
    seat_neighbors = {}
    for t, seats_per_side in tables.items():
        for row in [0, 1]:
            for col in range(seats_per_side):
                s = (t, row, col)
                neighbors = []
                if col - 1 >= 0:
                    neighbors.append((t, row, col - 1))
                if col + 1 < seats_per_side:
                    neighbors.append((t, row, col + 1))
                other = 1 - row
                neighbors.append((t, other, col))
                if col - 1 >= 0:
                    neighbors.append((t, other, col - 1))
                if col + 1 < seats_per_side:
                    neighbors.append((t, other, col + 1))
                seat_neighbors[s] = neighbors
    return seat_neighbors

#####################################
# 3. Optimization Functions          #
#####################################

def compute_cost(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                 neighbor_weight=1.0, corner_weight=3.0, gender_weight=5.0, fixed_weight=2.0, empty_weight=5.0):
    """
    Computes the total cost over all seating rounds.
    
    Cost components:
    1. **Neighbor Cost:** For each person (except those with neutral gender "X")
       the cost is (total neighbors encountered over rounds – number of unique neighbors).
       For fixed-seat persons, the cost is multiplied by fixed_weight.
    2. **Corner Cost:** For each person, if they sit in a corner (column 0 or last column)
       more than once, add (count - 1).
    3. **Gender Cost:** In each row of each table in each round, for every pair of adjacent seats
       with the same gender, add 1.
    4. **Empty Seat Clustering Cost:** In each row of each table in each round, for every adjacent pair
       where one seat is "Empty" and the other is not, add 1.
    
    The overall cost is the weighted sum of these components.
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
        for t, seats_per_side in tables.items():
            for row in [0, 1]:
                seats_in_row = [(t, row, col) for col in range(seats_per_side)]
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
                        
    empty_cost = 0
    for round_assign in assignments:
        for t, seats_per_side in tables.items():
            for row in [0, 1]:
                seats_in_row = [(t, row, col) for col in range(seats_per_side)]
                for i in range(len(seats_in_row) - 1):
                    p1 = round_assign[seats_in_row[i]]
                    p2 = round_assign[seats_in_row[i+1]]
                    # If one seat is empty and the other is not, add cost.
                    if (p1.startswith("Empty") and not p2.startswith("Empty")) or (not p1.startswith("Empty") and p2.startswith("Empty")):
                        empty_cost += 1
                        
    total_cost = (neighbor_weight * neighbor_cost +
                  corner_weight * corner_cost +
                  gender_weight * gender_cost +
                  empty_weight * empty_cost)
    return total_cost

def get_neighbors_info(assignments, seat_neighbors, tables):
    """Returns a dictionary mapping each person to the set of unique neighbors over all rounds."""
    person_neighbors = defaultdict(set)
    for round_assign in assignments:
        for seat, person in round_assign.items():
            for n_seat in seat_neighbors[seat]:
                neighbor_person = round_assign[n_seat]
                person_neighbors[person].add(neighbor_person)
    return person_neighbors

def initialize_assignments(people, tables, fixed_positions, num_rounds=3):
    """
    Creates an initial seating assignment for each round.
    Fixed seats (as specified in fixed_positions) are pre-assigned;
    the remaining seats are randomly filled with the remaining people.
    Returns a list (one per round) of dictionaries mapping seat -> person.
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
                         neighbor_weight=1.0, corner_weight=3.0, gender_weight=5.0, fixed_weight=2.0, empty_weight=5.0):
    """
    Uses simulated annealing to optimize seating assignments.
    In each iteration, two non-fixed seats in a random round are swapped.
    Moves that lower the total cost (or occasionally moves that worsen it) are accepted.
    Returns the best assignments found and the best cost.
    """
    current_cost = compute_cost(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                                neighbor_weight, corner_weight, gender_weight, fixed_weight, empty_weight)
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
                                neighbor_weight, corner_weight, gender_weight, fixed_weight, empty_weight)
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
# 4. Seating Arrangement Display     #
#####################################

def seating_dataframe_for_table(assignments, table_id, table_letter):
    """
    For the given table, sorts the seats (by row then column) and builds a DataFrame.
    Each row is labeled (e.g., A1, A2, …) and each column corresponds to an arrangement.
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

#####################################
# 5. Fixed Seat Input Parser         #
#####################################

def parse_fixed_seats(text):
    """
    Parses fixed seat assignments from text.
    Each line should be in the format: Name: SeatID (e.g., John: A12)
    Returns a dictionary mapping Name -> (table, row, col)
    """
    fixed = {}
    lines = text.strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        try:
            name_part, seat_part = line.split(":", 1)
            name = name_part.strip()
            seat_id = seat_part.strip().upper()
            
            # Extract table letter and seat number
            table_letter = seat_id[0]
            seat_num = int(seat_id[1:]) - 1  # Convert to 0-based index
            
            # Find table_id from table letter
            table_id = None
            for tid, letter in TABLE_LETTERS.items():
                if letter == table_letter:
                    table_id = tid
                    break
            if table_id is None:
                continue
                
            # Calculate row and column
            seats_per_side = TABLES[table_id]
            row = 1 if seat_num >= seats_per_side else 0
            col = seat_num % seats_per_side
            
            fixed[name] = (table_id, row, col)
        except Exception:
            continue
    return fixed

#####################################
# 6. Table Layout Visualization       #
#####################################

def generate_table_html(table_id, table_letter, tables):
    """
    Generates an HTML snippet representing the layout of one table.
    Seats are numbered sequentially (top row then bottom row).
    Corner seats are highlighted in light red.
    """
    num_cols = tables[table_id]
    top_row = [f"{table_letter}{i+1}" for i in range(num_cols)]
    bottom_row = [f"{table_letter}{i+1+num_cols}" for i in range(num_cols)]
    cell_style = (
        "width:40px; height:40px; border:1px solid #000; display:flex; "
        "align-items:center; justify-content:center; margin:2px; font-size:12px; font-weight:bold;"
    )
    corner_bg = "#ffcccc"  # light red for corners
    normal_bg = "#ffffff"
    def build_row_html(row_labels):
        row_html = "<div style='display:flex; justify-content:center;'>"
        for idx, label in enumerate(row_labels):
            bg = corner_bg if idx == 0 or idx == num_cols - 1 else normal_bg
            row_html += f"<div style='{cell_style} background-color:{bg};'>{label}</div>"
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

def display_table_layouts(tables, table_letters):
    st.markdown("## Table Layouts")
    st.markdown(
        """
        The following displays the seat numbering (blueprints) for each table.
        For example, Table A's seats are labeled A1, A2, … etc.
        Corner seats (first and last in each row) are highlighted in light red.
        """
    )
    for table_id in sorted(tables.keys()):
        table_letter = table_letters[table_id]
        html = generate_table_html(table_id, table_letter, tables)
        components.html(html, height=180, scrolling=False)

#####################################
# 7. Run Optimization & Build Data  #
#####################################

def run_optimization_and_build_data(iterations, initial_temp, cooling_rate,
                                      neighbor_weight, corner_weight, gender_weight, fixed_weight, empty_weight,
                                      people, person_genders, fixed_positions, tables):
    """
    Runs the seating optimization and returns:
      - best_assignments: list (per round) of seat -> person assignments,
      - best_cost: final cost,
      - neighbors_info: dict mapping each person to their overall set of neighbors.
    If there are fewer names than seats, fills remaining seats with "Empty" (gender "X").
    """
    total_seats = sum(2 * seats for seats in tables.values())
    if len(people) < total_seats:
        num_missing = total_seats - len(people)
        for i in range(num_missing):
            empty_name = f"Empty{i+1}"
            people.append(empty_name)
            person_genders[empty_name] = "X"
    elif len(people) > total_seats:
        people = people[:total_seats]
    seat_neighbors = compute_seat_neighbors(tables)
    assignments = initialize_assignments(people, tables, fixed_positions, num_rounds=3)
    best_assignments, best_cost = optimize_assignments(assignments, seat_neighbors, tables,
                                                       fixed_positions, person_genders,
                                                       iterations, initial_temp, cooling_rate,
                                                       neighbor_weight, corner_weight, gender_weight, fixed_weight, empty_weight)
    neighbors_info = get_neighbors_info(best_assignments, seat_neighbors, tables)
    return best_assignments, best_cost, neighbors_info

def combine_all_seating_dataframes(assignments, tables, table_letters):
    """
    Combines seating arrangements from all tables into a single DataFrame.
    Each row represents a seat, with columns for Table, Seat ID, and each arrangement round.
    """
    all_data = []
    for table_id in sorted(tables.keys()):
        table_letter = table_letters[table_id]
        seats = [seat for seat in assignments[0].keys() if seat[0] == table_id]
        seats.sort(key=lambda s: (s[1], s[2]))
        
        for seat in seats:
            row_data = {
                'Table': table_letter,
                'Seat ID': f"{table_letter}{seats.index(seat) + 1}"
            }
            for round_idx, assignment in enumerate(assignments):
                row_data[f'Arrangement {round_idx + 1}'] = assignment[seat]
            all_data.append(row_data)
    
    return pd.DataFrame(all_data)

#####################################
# 8. Main App                        #
#####################################

def get_current_settings():
    """Collects all current settings into a dictionary using current UI values."""
    # Get table definitions from text area
    table_def_text = st.session_state.table_def_text if 'table_def_text' in st.session_state else DEFAULT_TABLE_DEF
    
    # Get names from text areas
    default_male = "John\nMike\nDavid\nSteve\nRobert\nJames\nWilliam\nRichard\nJoseph\nThomas"
    default_female = "Mary\nLinda\nSusan\nKaren\nPatricia\nBarbara\nNancy\nLisa\nBetty\nMargaret"
    male_text = st.session_state.male_text if 'male_text' in st.session_state else default_male
    female_text = st.session_state.female_text if 'female_text' in st.session_state else default_female
    
    # Get fixed assignments
    fixed_text = st.session_state.fixed_text if 'fixed_text' in st.session_state else "John: A1\nMary: B2"
    
    # Get optimization parameters
    iterations = st.session_state.iterations if 'iterations' in st.session_state else 20000
    initial_temp = st.session_state.initial_temp if 'initial_temp' in st.session_state else 10.0
    cooling_rate = st.session_state.cooling_rate if 'cooling_rate' in st.session_state else 0.9995
    
    # Get weights
    neighbor_weight = st.session_state.neighbor_weight if 'neighbor_weight' in st.session_state else 1.0
    corner_weight = st.session_state.corner_weight if 'corner_weight' in st.session_state else 3.0
    gender_weight = st.session_state.gender_weight if 'gender_weight' in st.session_state else 5.0
    fixed_weight = st.session_state.fixed_weight if 'fixed_weight' in st.session_state else 2.0
    empty_weight = st.session_state.empty_weight if 'empty_weight' in st.session_state else 5.0
    
    return {
        "table_definitions": table_def_text,
        "male_names": male_text,
        "female_names": female_text,
        "fixed_assignments": fixed_text,
        "optimization_params": {
            "iterations": iterations,
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
        },
        "weights": {
            "neighbor_weight": neighbor_weight,
            "corner_weight": corner_weight,
            "gender_weight": gender_weight,
            "fixed_weight": fixed_weight,
            "empty_weight": empty_weight
        }
    }

def main():
    st.title("SeatPlan")
    st.markdown("##### Optimizing seating arrangements for events.")
    st.markdown("###### Use sidebar for additional customization")

    
    # Add download button in the sidebar at the top
    st.sidebar.markdown("# Settings Import/Export")
    
    # Download and upload buttons remain in sidebar
    if st.sidebar.button("Download Current Settings"):
        settings = get_current_settings()
        settings_json = json.dumps(settings, indent=2)
        st.sidebar.write("Settings ready for download:")
        st.sidebar.json(settings)  # Display the settings to verify content
        download = st.sidebar.download_button(
            label="👉 Click here to download Settings JSON",
            data=settings_json,
            file_name="seatplan_settings.json",
            mime="application/json",
            key="settings_download"  # Add a unique key
        )
    
    uploaded_file = st.sidebar.file_uploader("Upload Settings", type=['json'])
    if uploaded_file is not None:
        try:
            settings = json.load(uploaded_file)
            st.session_state.uploaded_settings = settings
            st.sidebar.success("Settings loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading settings: {str(e)}")
    
    st.sidebar.markdown("# Conditions")
    
    # Move table definitions to main content area
    st.header("Table Layout Configuration")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if "uploaded_settings" in st.session_state:
            settings = st.session_state.uploaded_settings
            table_def_text = st.text_area(
                "Define tables (e.g., 'A: 8')", 
                value=settings["table_definitions"],
                height=150,
                key='table_def_text',
                on_change=lambda: None,  # This ensures the session state is updated
                help="Each line should be in the format 'Letter: Number', where Number is the number of seats per side."
            )
        else:
            table_def_text = st.text_area(
                "Define tables (e.g., 'A: 8')", 
                value=DEFAULT_TABLE_DEF,
                height=150,
                key='table_def_text',
                on_change=lambda: None,  # This ensures the session state is updated
                help="Each line should be in the format 'Letter: Number', where Number is the number of seats per side."
            )
        
    global TABLES, TABLE_LETTERS
    TABLES, TABLE_LETTERS = parse_table_definitions(table_def_text)
        

    with col2:
        # Show table layouts inline
        st.markdown(
            """
            Seat numberings for each table.
            Corner seats are highlighted in light red.
            """
        )
        for table_id in sorted(TABLES.keys()):
            table_letter = TABLE_LETTERS[table_id]
            html = generate_table_html(table_id, table_letter, TABLES)
            components.html(html, height=180, scrolling=False)

    st.sidebar.header("Name Lists")
    if "uploaded_settings" in st.session_state:
        settings = st.session_state.uploaded_settings
        male_text = st.sidebar.text_area("Male Names (one per line)", 
                                     value=settings["male_names"], height=150,
                                     key='male_text',
                                     on_change=lambda: None,
                                     help="Enter one male name per line.")
        female_text = st.sidebar.text_area("Female Names (one per line)", 
                                       value=settings["female_names"], height=150,
                                       key='female_text',
                                       on_change=lambda: None,
                                       help="Enter one female name per line.")
    else:
        default_male = "John\nMike\nDavid\nSteve\nRobert\nJames\nWilliam\nRichard\nJoseph\nThomas"
        default_female = "Mary\nLinda\nSusan\nKaren\nPatricia\nBarbara\nNancy\nLisa\nBetty\nMargaret"
        male_text = st.sidebar.text_area("Male Names (one per line)", value=default_male, height=150,
                                         key='male_text',
                                         on_change=lambda: None,
                                         help="Enter one male name per line.")
        female_text = st.sidebar.text_area("Female Names (one per line)", value=default_female, height=150,
                                           key='female_text',
                                           on_change=lambda: None,
                                           help="Enter one female name per line.")
    male_names = [name.strip() for name in male_text.splitlines() if name.strip()]
    female_names = [name.strip() for name in female_text.splitlines() if name.strip()]
    people = male_names + female_names
    person_genders = {}
    for name in male_names:
        person_genders[name] = "M"
    for name in female_names:
        person_genders[name] = "F"
    
    st.sidebar.header("Fixed Seat Assignments")
    if "uploaded_settings" in st.session_state:
        settings = st.session_state.uploaded_settings
        fixed_text = st.sidebar.text_area("Enter fixed assignments (e.g., 'John: A12')", 
                                          value=settings["fixed_assignments"], height=100,
                                          key='fixed_text',
                                          on_change=lambda: None,
                                          help="Each line should be in the format 'Name: Seat' (e.g., 'John: A12'). Ensure that the seat exists in the overview")
    else:
        fixed_text = st.sidebar.text_area("Enter fixed assignments (e.g., 'John: A12')", 
                                          value="John: A1\nMary: B2", height=100,
                                          key='fixed_text',
                                          on_change=lambda: None,
                                          help="Each line should be in the format 'Name: Seat' (e.g., 'John: A12'). Ensure that the seat exists in the overview")
    fixed_positions = parse_fixed_seats(fixed_text)
    

    st.sidebar.header("Condition Weights")
    st.sidebar.markdown("""
        Importance of a condition. The accumulative sum of all conditions is optimized to be as low as possible. HIGHER more important. 
    """)
    

    if "uploaded_settings" in st.session_state:
        settings = st.session_state.uploaded_settings
        neighbor_weight = st.sidebar.number_input("Neighbor Weight", 
                                             value=settings["weights"]["neighbor_weight"], 
                                             step=0.1, format="%.1f",
                                             key='neighbor_weight',
                                             on_change=lambda: None,
                                             help="Weight for penalizing repeated neighbors. Higher values force more neighbor diversity.")
        corner_weight = st.sidebar.number_input("Corner Weight", 
                                           value=settings["weights"]["corner_weight"], 
                                           step=0.1, format="%.1f",
                                           key='corner_weight',
                                           on_change=lambda: None,
                                           help="Weight for penalizing repeated corner seatings.")
        gender_weight = st.sidebar.number_input("Gender Weight", 
                                           value=settings["weights"]["gender_weight"], 
                                           step=0.1, format="%.1f",
                                           key='gender_weight',
                                           on_change=lambda: None,
                                           help="Weight for penalizing adjacent seats with the same gender in a row.")
        fixed_weight = st.sidebar.number_input("Fixed Seat Diversity Weight", 
                                          value=settings["weights"]["fixed_weight"], 
                                          step=0.1, format="%.1f",
                                          key='fixed_weight',
                                          on_change=lambda: None,
                                          help="Extra weight applied to fixed-seat persons to encourage diverse neighbors.")
        empty_weight = st.sidebar.number_input("Empty Seat Clustering Weight", 
                                          value=settings["weights"]["empty_weight"], 
                                          step=0.1, format="%.1f",
                                          key='empty_weight',
                                          on_change=lambda: None,
                                          help="Weight for penalizing boundaries between empty and non-empty seats. Higher values encourage empty seats to cluster together.")
    else:
        neighbor_weight = st.sidebar.number_input("Neighbor", value=1.0, step=0.1, format="%.1f",
                                                  key='neighbor_weight',
                                                  on_change=lambda: None,
                                                  help="Weight for penalizing repeated neighbors. Higher values force more neighbor diversity.")
        corner_weight = st.sidebar.number_input("Corner", value=3.0, step=0.1, format="%.1f",
                                                key='corner_weight',
                                                on_change=lambda: None,
                                                help="Weight for penalizing repeated corner seatings.")
        gender_weight = st.sidebar.number_input("Gender", value=5.0, step=0.1, format="%.1f",
                                                key='gender_weight',
                                                on_change=lambda: None,
                                                help="Weight for penalizing adjacent seats with the same gender in a row.")
        fixed_weight = st.sidebar.number_input("Fixed Seat Diversity", value=2.0, step=0.1, format="%.1f",
                                               key='fixed_weight',
                                               on_change=lambda: None,
                                               help="Extra weight applied to fixed-seat persons to encourage diverse neighbors.")
        empty_weight = st.sidebar.number_input("Empty Seat Clustering", value=5.0, step=0.1, format="%.1f",
                                               key='empty_weight',
                                               on_change=lambda: None,
                                               help="Weight for penalizing boundaries between empty and non-empty seats. Higher values encourage empty seats to cluster together.")
    
    # Replace the header with an expander
    with st.sidebar.expander("Optimization Parameters", expanded=False):
        if "uploaded_settings" in st.session_state:
            settings = st.session_state.uploaded_settings
            iterations = st.number_input("Iterations", 
                                    value=settings["optimization_params"]["iterations"], 
                                    step=1000, min_value=1000,
                                    key='iterations',
                                    on_change=lambda: None,
                                    help="Number of iterations for simulated annealing. More iterations may yield better results but take longer.")
            initial_temp = st.number_input("Initial Temperature", 
                                      value=settings["optimization_params"]["initial_temp"], 
                                      step=1.0,
                                      key='initial_temp',
                                      on_change=lambda: None,
                                      help="The starting temperature for simulated annealing. Higher values allow more exploration.")
            cooling_rate = st.slider("Cooling Rate", min_value=0.990, max_value=0.9999, 
                                value=settings["optimization_params"]["cooling_rate"],
                                key='cooling_rate',
                                on_change=lambda: None,
                                help="The multiplier applied to the temperature after each iteration. Values closer to 1.0 cool more slowly.")
        else:
            iterations = st.number_input("Iterations", value=20000, step=1000, min_value=1000,
                                       key='iterations',
                                       on_change=lambda: None,
                                       help="Number of iterations for simulated annealing. More iterations may yield better results but take longer.")
            initial_temp = st.number_input("Initial Temperature", value=10.0, step=1.0,
                                       key='initial_temp',
                                       on_change=lambda: None,
                                       help="The starting temperature for simulated annealing. Higher values allow more exploration.")
            cooling_rate = st.slider("Cooling Rate", min_value=0.990, max_value=0.9999, value=0.9995,
                                 key='cooling_rate',
                                 on_change=lambda: None,
                                 help="The multiplier applied to the temperature after each iteration. Values closer to 1.0 cool more slowly.")
    
    run_button = st.sidebar.button("Run Optimization")
    
    if run_button or "best_assignments" not in st.session_state:
        with st.spinner("Running optimization..."):
            best_assignments, best_cost, neighbors_info = run_optimization_and_build_data(
                iterations, initial_temp, cooling_rate,
                neighbor_weight, corner_weight, gender_weight, fixed_weight, empty_weight,
                people, person_genders, fixed_positions, TABLES
            )
            st.session_state.best_assignments = best_assignments
            st.session_state.best_cost = best_cost
            st.session_state.neighbors_info = neighbors_info
            st.session_state.person_genders = person_genders
            
            # Create combined DataFrame and store in session state
            combined_df = combine_all_seating_dataframes(best_assignments, TABLES, TABLE_LETTERS)
            st.session_state.combined_df = combined_df
    
    st.success(f"Optimization complete. Best cost: {st.session_state.best_cost}")
    


    st.header("Seating Arrangements")
    # Add download button for combined seating arrangements
    st.download_button(
        label="Download All Seating Arrangements",
        data=st.session_state.combined_df.to_csv(index=False),
        file_name="seating_arrangements.csv",
        mime="text/csv"
    )
    for table_id in sorted(TABLES.keys()):
        table_letter = TABLE_LETTERS[table_id]
        df = seating_dataframe_for_table(st.session_state.best_assignments, table_id, table_letter)
        st.subheader(f"Table {table_letter}")
        st.dataframe(df, height=300)
    
    st.header("Overall Neighbour Summary")
    data = []
    for person, neighbours in st.session_state.neighbors_info.items():
        data.append((person, st.session_state.person_genders.get(person, "X"), ", ".join(sorted(neighbours))))
    nbr_df = pd.DataFrame(data, columns=["Person", "Gender", "Neighbours"])
    st.dataframe(nbr_df, height=400)


if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
