import streamlit as st
import streamlit.web.bootstrap
import streamlit.components.v1 as components
import pandas as pd
import random
import math
import copy
from collections import defaultdict
import json
from enum import Enum

class Settings(Enum):
    TABLE_DEFINITIONS = "table_definitions"
    MALES = "males"
    FEMALES = "females"
    FIXED_ASSIGNMENTS = "fixed_assignments"
    PREFERRED_SIDE_PREFERENCES = "preferred_side_preferences"
    EXCLUDED_PREFERENCES = "excluded_preferences"

    def __str__(self):
        return self.value

DEFAULT_SETTINGS = json.load(open("default_settings.json", "r", encoding="utf-8"))


##### 

def compute_seat_neighbors(tables):
    seat_neighbors = {}
    for t, seats_per_side in tables.items():
        for row in [0, 1]:
            for col in range(seats_per_side):
                s = (t, row, col)
                neighbors = {
                    "side": [],
                    "front": [],
                    "diagonal": [],
                    "all": []
                }
                # Side neighbours
                if col - 1 >= 0:
                    neighbors["side"].append((t, row, col - 1))
                if col + 1 < seats_per_side:
                    neighbors["side"].append((t, row, col + 1))
                # Front neighbour (directly opposite)
                other = 1 - row
                neighbors["front"].append((t, other, col))
                # Diagonal neighbours
                if col - 1 >= 0:
                    neighbors["diagonal"].append((t, other, col - 1))
                if col + 1 < seats_per_side:
                    neighbors["diagonal"].append((t, other, col + 1))
                seat_neighbors[s] = neighbors
                neighbors["all"] = neighbors["side"] + neighbors["front"] + neighbors["diagonal"]
    return seat_neighbors

def parse_table_definitions(text):
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

def parse_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def get_setting(settings_dict, setting_key):
    """Helper function to get a setting value using an enum key"""
    return settings_dict[str(setting_key)]

def generate_seats(tables):
    seats = []
    for t, seats_per_side in tables.items():
        for row in [0, 1]:
            for col in range(seats_per_side):
                seats.append((t, row, col))
    return seats




##### UI

def generate_arrangement_html(arrangement, table_id, tables):
    """
    Generates an HTML representation of one table for one seating arrangement.
    - arrangement: dict mapping seat (table, row, col) to occupant name.

    """
    num_cols = tables[table_id]
    cell_style = (
        "width:60px; height:60px; border:1px solid #000; display:flex; "
        "align-items:center; justify-content:center; margin:1px; font-size:12px; font-weight:bold;"
        "text-align:center; word-break:break-word; overflow:hidden;"
    )
    def get_bg_color():
        return "#ffffff"
        
    top_html = "<div style='display:flex; justify-content:center; flex-wrap:nowrap;'>"
    for col in range(num_cols):
        seat = (table_id, 0, col)
        occupant = arrangement.get(seat, "")
        bg_color = get_bg_color()
        top_html += f"<div style='{cell_style} background-color:{bg_color};'>{occupant}</div>"
    top_html += "</div>"
    
    bottom_html = "<div style='display:flex; justify-content:center; flex-wrap:nowrap;'>"
    for col in range(num_cols):
        seat = (table_id, 1, col)
        occupant = arrangement.get(seat, "")
        bg_color = get_bg_color()
        bottom_html += f"<div style='{cell_style} background-color:{bg_color};'>{occupant}</div>"
    bottom_html += "</div>"
    
    full_html = f"""
    <html>
      <head>
        <meta charset="UTF-8">
        <style>
          body {{ font-family: sans-serif; margin:5px; padding:0; }}
        </style>
      </head>
      <body>
        {top_html}
        {bottom_html}
      </body>
    </html>
    """
    return full_html

def generate_arrangement_html(arrangement, table_id, tables):
    num_cols = tables[table_id]
    cell_style = (
        "width:60px; height:60px; border:1px solid #000; display:flex; "
        "align-items:center; justify-content:center; margin:1px; font-size:12px; font-weight:bold;"
        "text-align:center; word-break:break-word; overflow:hidden;"
    )
    def get_bg_color():
        return "#ffffff"
    
    # Use flex with no wrapping and enable horizontal scrolling
    container_style = "display:flex; justify-content:center; flex-wrap:nowrap; overflow-x:auto;"
    
    top_html = f"<div style='{container_style}'>"
    for col in range(num_cols):
        seat = (table_id, 0, col)
        occupant = arrangement.get(seat, "")
        bg_color = get_bg_color()
        top_html += f"<div style='{cell_style} background-color:{bg_color};'>{occupant}</div>"
    top_html += "</div>"
    
    bottom_html = f"<div style='{container_style}'>"
    for col in range(num_cols):
        seat = (table_id, 1, col)
        occupant = arrangement.get(seat, "")
        bg_color = get_bg_color()
        bottom_html += f"<div style='{cell_style} background-color:{bg_color};'>{occupant}</div>"
    bottom_html += "</div>"
    
    full_html = f"""
    <html>
      <head>
        <meta charset="UTF-8">
        <style>
          body {{ font-family: sans-serif; margin:5px; padding:0; }}
        </style>
      </head>
      <body>
        {top_html}
        {bottom_html}
      </body>
    </html>
    """
    return full_html


def generate_table_html(table_id, table_letter, tables):
    num_cols = tables[table_id]
    top_row = [f"{table_letter}{i+1}" for i in range(num_cols)]
    bottom_row = [f"{table_letter}{i+1+num_cols}" for i in range(num_cols)]
    cell_style = (
        "width:40px; height:40px; border:1px solid #000; display:flex; "
        "align-items:center; justify-content:center; margin:2px; font-size:12px; font-weight:bold;"
    )
    corner_bg = "#ffcccc"  # light red
    normal_bg = "#ffffff"
    def build_row_html(row_labels):
        row_html = "<div style='display:flex; justify-content:center; flex-wrap:wrap;'>"
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
        {top_html}
        {bottom_html}
      </body>
    </html>
    """
    return full_html

## Input

def load_settings():
    """Load settings from default or uploaded file"""
    if "uploaded_settings" in st.session_state:
        return st.session_state.uploaded_settings
    else:
        return DEFAULT_SETTINGS

def save_current_settings():
    """Save the current settings to a JSON file"""
    # Collect all current settings
    current_settings = {}
    for setting in Settings:
        setting_key = str(setting)
        if setting_key in st.session_state:
            current_settings[setting_key] = st.session_state[setting_key]
    
    # Convert to JSON string with proper formatting
    settings_json = json.dumps(current_settings, indent=2)
    
    # Show a preview of the settings
    st.code(settings_json, language="json")
    
    # Create a download button
    st.download_button(
        label="Download Settings",
        data=settings_json,
        file_name="seatplan_settings.json",
        mime="application/json"
    )

def import_settings():
    """Import settings from a JSON file"""
    uploaded_file = st.file_uploader("Upload settings file", type=["json"])
    
    if uploaded_file is not None:
        try:
            settings = json.load(uploaded_file)
            st.session_state.uploaded_settings = settings
            st.success("Settings loaded successfully!")
            # Force a rerun to apply the new settings
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")

def set_guests():
    with st.sidebar.expander("Guests"):
        st.markdown("""
            Enter the names of the guests.
        """)
        
        settings = load_settings()
        
        male_text = st.text_area("Males (one per line)", 
                                  value=get_setting(settings, Settings.MALES), 
                                  height=150,
                                  key=str(Settings.MALES), 
                                  help="One male name per line.")
        male_count = len([name for name in male_text.splitlines() if name.strip()])
        st.caption(f"Male count: {male_count}")
        
        female_text = st.text_area("Females (one per line)", 
                                   value=get_setting(settings, Settings.FEMALES), 
                                   height=150,
                                   key=str(Settings.FEMALES), 
                                   help="One female name per line.")
        female_count = len([name for name in female_text.splitlines() if name.strip()])
        st.caption(f"Female count: {female_count}")
        
        male_names = parse_lines(male_text)
        female_names = parse_lines(female_text)
        guests = male_names + female_names
        
        # Check for duplicates
        duplicate_names = [name for name in set(guests) if guests.count(name) > 1]
        if duplicate_names:
            st.error(f"Duplicate names found: {', '.join(duplicate_names)}")
            
        person_genders = {}
        for name in male_names:
            person_genders[name] = "M"
        for name in female_names:
            person_genders[name] = "F"
        st.session_state.guests = guests

def set_tables():
    # Table Layout Configuration
    with st.expander("Table Configurations"):
        col1, col2 = st.columns([1, 2])
        with col1:
            settings = load_settings()
            st.text_area(
                "Define tables (e.g., 'A: 8')", 
                value=get_setting(settings, Settings.TABLE_DEFINITIONS),
                height=150,
                key=str(Settings.TABLE_DEFINITIONS),
                help="Each line: Letter: Number"
            )
                
        TABLES, TABLE_LETTERS = parse_table_definitions(st.session_state[str(Settings.TABLE_DEFINITIONS)])
        
        with col2:
            for table_id in sorted(TABLES.keys()):
                table_letter = TABLE_LETTERS[table_id]
                html = generate_table_html(table_id, table_letter, TABLES)
                # Calculate dynamic height based on number of seats
                table_height = max(150, 100 + (TABLES[table_id] // 8) * 50)
                components.html(html, height=table_height, scrolling=True)

def show_arrangements(arrangements, tables, table_letters):    
    st.markdown("## Arrangements") 
    
    cols = st.columns([1, 1, 5])  
    with cols[0]:
        tables_per_row = st.number_input("Tables per row", min_value=1, max_value=5, value=3, step=1)
    
    # Display each arrangement
    for round_idx, arrangement in enumerate(arrangements):
        st.markdown(f"### Arrangement {round_idx + 1}")
        
        # Create columns for each table
        cols = st.columns(len(tables))
        
        # Display each table
        table_ids = sorted(tables.keys())

        for i in range(0, len(table_ids), tables_per_row):
            cols = st.columns(tables_per_row)
            for j, table_id in enumerate(table_ids[i:i+tables_per_row]):
                with cols[j]:
                    table_letter = table_letters[table_id]
                    st.markdown(f"**Table {table_letter}**")
                    html = generate_arrangement_html(arrangement, table_id, tables)
                    # Calculate dynamic height and width based on number of seats
                    table_height = max(150, 100 + (tables[table_id] // 8) * 50)
                    calculated_width = tables[table_id] * 62  # Adjust this multiplier if needed
                    components.html(html, height=table_height, width=calculated_width, scrolling=True)


def seat_label_to_tuple(seat_label, TABLES, TABLE_LETTERS):
    """
    Convert a seat label (like "A1" or "A9") into a seat tuple (table_id, row, col)
    based on TABLES (mapping table_id -> seats_per_side) and TABLE_LETTERS (mapping table_id -> letter).
    """
    for table_id, letter in TABLE_LETTERS.items():
        if seat_label.startswith(letter):
            try:
                num = int(seat_label[len(letter):])
            except ValueError:
                continue
            seats_per_side = TABLES[table_id]
            if num <= seats_per_side:
                row = 0
                col = num - 1
            else:
                row = 1
                col = num - seats_per_side - 1
            return (table_id, row, col)
    return None

def run_optimization(tables, guests, num_rounds=3, locked_seats_per_round=None, table_letters=None, 
                    previous_arrangements=None, person_genders=None, preferred_neighbors=None, 
                    excluded_neighbors=None, max_iterations=1000):
    """
    Args:
        tables: {table_id: seats_per_side}
        guests: list of guest names
        num_rounds: number of rounds
        locked_seats_per_round: {round_index: {seat_label: guest_name}}
        previous_arrangements: list of previous arrangements to build upon
        person_genders: {guest_name: "M"/"F"} for gender-alternating seating
        preferred_neighbors: {guest_name: [preferred_guest_names]}
        excluded_neighbors: {guest_name: [excluded_guest_names]}
        max_iterations: maximum number of optimization iterations
    Returns:
        [{(table_id, row, col): guest_name}]
    """
    # Generate all seats and compute neighbors
    seats = generate_seats(tables)
    seat_neighbors = compute_seat_neighbors(tables)
    
    # Initialize arrangements
    arrangements = []
    if previous_arrangements:
        # Start with previous arrangements if provided
        arrangements = copy.deepcopy(previous_arrangements)
        # Only generate additional rounds if needed
        num_rounds = max(0, num_rounds - len(arrangements))
    
    # Generate initial arrangements for new rounds
    for round_index in range(num_rounds):
        # Initialize with locked seats
        arrangement = {}
        if locked_seats_per_round and round_index in locked_seats_per_round:
            locked_assignments = locked_seats_per_round[round_index]
            for seat_label, guest in locked_assignments.items():
                seat_tuple = seat_label_to_tuple(seat_label, tables, table_letters)
                if seat_tuple:
                    arrangement[seat_tuple] = guest
        
        # Get locked guests and available guests
        locked_guests = set(arrangement.values())
        available_guests = [g for g in guests if g not in locked_guests]
        
        # Initial random placement
        random.shuffle(available_guests)
        for seat in seats:
            if seat not in arrangement:
                if available_guests:
                    arrangement[seat] = available_guests.pop(0)
                else:
                    break
        
        arrangements.append(arrangement)
    
    # Now optimize all arrangements together
    if max_iterations > 0:
        arrangements = optimize_all_arrangements(
            arrangements, 
            seats, 
            tables,
            table_letters,
            seat_neighbors, 
            person_genders, 
            preferred_neighbors, 
            excluded_neighbors,
            locked_seats_per_round,
            max_iterations
        )
    
    return arrangements

def optimize_all_arrangements(arrangements, seats, tables, table_letters, seat_neighbors, person_genders, 
                             preferred_neighbors, excluded_neighbors, locked_seats_per_round,
                             max_iterations):
    """
    Optimize all seating arrangements together, considering:
    1. Alternating male/female within each arrangement
    2. Preferred neighbors within each arrangement
    3. Excluded neighbors within each arrangement
    4. Avoiding repeat neighbors across different arrangements
    
    Uses a local search algorithm with random swaps and flips.
    """
    # Weights for different optimization criteria
    GENDER_ALTERNATION_WEIGHT = 15  # Weight for gender alternation
    PREFERRED_NEIGHBOR_WEIGHT = 20  # Weight for preferred neighbors
    EXCLUDED_NEIGHBOR_WEIGHT = 50   # Weight for excluded neighbors
    REPEAT_NEIGHBOR_WEIGHT = 15     # Weight for repeat neighbors across rounds
    
    # Create reverse mappings from person to seat for each arrangement
    person_to_seat_by_round = []
    for arrangement in arrangements:
        person_to_seat = {person: seat for seat, person in arrangement.items()}
        person_to_seat_by_round.append(person_to_seat)
    
    # Create a placeholder for the progress chart
    progress_chart = st.empty()
    
    # Create containers for detailed cost breakdown
    cost_breakdown = st.expander("Cost Function Breakdown", expanded=False)
    
    # Create container for problematic guests (only used at the end)
    problematic_guests = st.expander("Problematic Guests", expanded=False)
    
    # Lists to track progress
    iterations = []
    best_scores = []  # Only track best scores
    component_scores = {
        "Gender Alternation": [],
        "Preferred Neighbors": [],
        "Excluded Neighbors": [],
        "Repeat Neighbors": []
    }
    
    # Define the scoring function for all arrangements
    def score_all_arrangements(collect_components=False, arrangements_to_score=None, track_guest_costs=False):
        if arrangements_to_score is None:
            arrangements_to_score = arrangements
            
        total_score = 0
        
        # Component scores if we're collecting them
        gender_score = 0
        preferred_score = 0
        excluded_score = 0
        repeat_score = 0
        
        # Track costs per guest if requested
        guest_costs = defaultdict(lambda: defaultdict(int)) if track_guest_costs else None
        
        # Track all neighbor pairs across all rounds
        all_neighbor_pairs = set()
        
        # Score each arrangement individually
        for round_idx, arrangement in enumerate(arrangements_to_score):
            round_score = 0
            
            # Score gender alternation
            if person_genders:
                for seat in seats:
                    if seat in arrangement:
                        person = arrangement[seat]
                        person_gender = person_genders.get(person)
                        
                        # Check side neighbors (left/right)
                        for neighbor_seat in seat_neighbors[seat]["side"]:
                            if neighbor_seat in arrangement:
                                neighbor = arrangement[neighbor_seat]
                                neighbor_gender = person_genders.get(neighbor)
                                
                                # Penalize same gender neighbors on the sides
                                if person_gender == neighbor_gender:
                                    penalty = -GENDER_ALTERNATION_WEIGHT
                                    round_score += penalty
                                    if collect_components:
                                        gender_score += penalty
                                    if track_guest_costs:
                                        guest_costs[person]["gender"] += penalty / 2
                                        guest_costs[neighbor]["gender"] += penalty / 2
                                else:
                                    bonus = GENDER_ALTERNATION_WEIGHT / 2
                                    round_score += bonus
                                    if collect_components:
                                        gender_score += bonus
                                    if track_guest_costs:
                                        guest_costs[person]["gender"] += bonus / 2
                                        guest_costs[neighbor]["gender"] += bonus / 2
                        
                        # Check opposite neighbors (across the table)
                        for neighbor_seat in seat_neighbors[seat]["front"]:
                            if neighbor_seat in arrangement:
                                neighbor = arrangement[neighbor_seat]
                                neighbor_gender = person_genders.get(neighbor)
                                
                                # Penalize same gender neighbors across the table
                                if person_gender == neighbor_gender:
                                    penalty = -GENDER_ALTERNATION_WEIGHT
                                    round_score += penalty
                                    if collect_components:
                                        gender_score += penalty
                                    if track_guest_costs:
                                        guest_costs[person]["gender"] += penalty / 2
                                        guest_costs[neighbor]["gender"] += penalty / 2
                                else:
                                    bonus = GENDER_ALTERNATION_WEIGHT / 2
                                    round_score += bonus
                                    if collect_components:
                                        gender_score += bonus
                                    if track_guest_costs:
                                        guest_costs[person]["gender"] += bonus / 2
                                        guest_costs[neighbor]["gender"] += bonus / 2
            
            # Score preferred neighbors
            if preferred_neighbors:
                for seat in seats:
                    if seat in arrangement:
                        person = arrangement[seat]
                        if person in preferred_neighbors:
                            for neighbor_seat in seat_neighbors[seat]["all"]:
                                if neighbor_seat in arrangement:
                                    neighbor = arrangement[neighbor_seat]
                                    # Reward preferred neighbors
                                    if neighbor in preferred_neighbors.get(person, []):
                                        bonus = PREFERRED_NEIGHBOR_WEIGHT
                                        round_score += bonus
                                        if collect_components:
                                            preferred_score += bonus
                                        if track_guest_costs:
                                            guest_costs[person]["preferred"] += bonus / 2
                                            guest_costs[neighbor]["preferred"] += bonus / 2
            
            # Score excluded neighbors
            if excluded_neighbors:
                for seat in seats:
                    if seat in arrangement:
                        person = arrangement[seat]
                        if person in excluded_neighbors:
                            for neighbor_seat in seat_neighbors[seat]["all"]:
                                if neighbor_seat in arrangement:
                                    neighbor = arrangement[neighbor_seat]
                                    # Heavily penalize excluded neighbors
                                    if neighbor in excluded_neighbors.get(person, []):
                                        penalty = -EXCLUDED_NEIGHBOR_WEIGHT
                                        round_score += penalty
                                        if collect_components:
                                            excluded_score += penalty
                                        if track_guest_costs:
                                            guest_costs[person]["excluded"] += penalty / 2
                                            guest_costs[neighbor]["excluded"] += penalty / 2
            
            # Collect neighbor pairs for this round
            for seat in seats:
                if seat in arrangement:
                    person = arrangement[seat]
                    for neighbor_seat in seat_neighbors[seat]["all"]:
                        if neighbor_seat in arrangement:
                            neighbor = arrangement[neighbor_seat]
                            # Store the neighbor pair (sorted to avoid duplicates)
                            pair = tuple(sorted([person, neighbor]))
                            # Add to the set with round information
                            all_neighbor_pairs.add((pair, round_idx))
            
            total_score += round_score
        
        # Penalize repeat neighbors across different rounds
        neighbor_count = defaultdict(list)
        for (pair, round_idx) in all_neighbor_pairs:
            neighbor_count[pair].append(round_idx)
        
        for pair, rounds in neighbor_count.items():
            if len(rounds) > 1:
                # Apply increasing penalty for each repeat
                repeat_penalty = sum(range(len(rounds))) * REPEAT_NEIGHBOR_WEIGHT
                total_score -= repeat_penalty
                if collect_components:
                    repeat_score -= repeat_penalty
                if track_guest_costs:
                    person1, person2 = pair
                    penalty_per_person = -repeat_penalty / 2
                    guest_costs[person1]["repeat"] += penalty_per_person
                    guest_costs[person2]["repeat"] += penalty_per_person
        
        if track_guest_costs:
            # Calculate total cost per guest
            for person in guest_costs:
                guest_costs[person]["total"] = sum(guest_costs[person].values())
            
            if collect_components:
                return total_score, {
                    "Gender Alternation": gender_score,
                    "Preferred Neighbors": preferred_score,
                    "Excluded Neighbors": excluded_score,
                    "Repeat Neighbors": repeat_score
                }, guest_costs
            return total_score, guest_costs
        
        if collect_components:
            return total_score, {
                "Gender Alternation": gender_score,
                "Preferred Neighbors": preferred_score,
                "Excluded Neighbors": excluded_score,
                "Repeat Neighbors": repeat_score
            }
        return total_score
    
    # Identify locked seats and people for each round
    locked_seats_by_round = []
    locked_people_by_round = []
    
    for round_idx in range(len(arrangements)):
        locked_seats = set()
        locked_people = set()
        
        if locked_seats_per_round and round_idx in locked_seats_per_round:
            for seat_label, person in locked_seats_per_round[round_idx].items():
                seat_tuple = seat_label_to_tuple(seat_label, tables, table_letters)
                if seat_tuple:
                    locked_seats.add(seat_tuple)
                    locked_people.add(person)
        
        locked_seats_by_round.append(locked_seats)
        locked_people_by_round.append(locked_people)
    
    # Identify free seats for each round (seats that aren't locked)
    free_seats_by_round = []
    for round_idx in range(len(arrangements)):
        free_seats_by_round.append([s for s in seats if s not in locked_seats_by_round[round_idx]])
    
    # Initial score with component breakdown and guest costs
    initial_score, components, guest_costs = score_all_arrangements(collect_components=True, track_guest_costs=True)
    
    # Store the best solution found so far
    best_score = initial_score
    best_arrangements = copy.deepcopy(arrangements)
    best_guest_costs = guest_costs
    
    # Current working solution (may be worse than best)
    current_score = initial_score
    current_arrangements = copy.deepcopy(arrangements)
    
    # Record initial scores
    iterations.append(0)
    best_scores.append(best_score)
    for component, value in components.items():
        component_scores[component].append(value)
    
    # Simulated annealing parameters
    temperature = 1.0
    cooling_rate = 0.995
    
    # Update frequency (how often to update the chart)
    update_freq = max(1, max_iterations // 100)
    
    # Probability of trying different move types
    flip_probability = 0.2  # 20% chance for table corner flip
    adjacent_swap_probability = 0.3  # 30% chance for adjacent swap
    # Remaining 50% will be random swaps
    
    # Optimization loop
    for iteration in range(max_iterations):
        # Work with a copy of the current arrangements
        working_arrangements = copy.deepcopy(current_arrangements)
        
        # Randomly select a round
        round_idx = random.randrange(len(working_arrangements))
        
        arrangement = working_arrangements[round_idx]
        
        # Get locked seats and people for this round
        locked_seats = locked_seats_by_round[round_idx]
        locked_people = locked_people_by_round[round_idx]
        
        # Decide which move type to try
        move_type_rand = random.random()
        
        if move_type_rand < flip_probability:
            # ------------------ FLIP MOVE ------------------
            # Choose a random table
            table_ids = list(tables.keys())
            if not table_ids:
                continue
            table_id = random.choice(table_ids)
            seats_per_side = tables[table_id]
            
            # Get free seats for this table in the current round
            free_seats_table = [s for s in free_seats_by_round[round_idx] if s[0] == table_id]
            
            # Choose a corner at random: "left" or "right"
            corner = random.choice(["left", "right"])
            
            # Define a helper to compute the maximum contiguous block width from a given corner
            def max_block_width_for_row(row, corner):
                width = 0
                if corner == "left":
                    for col in range(seats_per_side):
                        seat = (table_id, row, col)
                        if seat in free_seats_table:
                            width += 1
                        else:
                            break
                else:  # "right" corner: start at the rightmost column and go left
                    for col in range(seats_per_side - 1, -1, -1):
                        seat = (table_id, row, col)
                        if seat in free_seats_table:
                            width += 1
                        else:
                            break
                return width
            
            # Calculate maximum contiguous block width for each row
            max_width_row0 = max_block_width_for_row(0, corner)
            max_width_row1 = max_block_width_for_row(1, corner)
            max_width = min(max_width_row0, max_width_row1)
            
            # If no contiguous free block exists, fall back to swap move
            if max_width < 1:
                # Fall back to swap move
                if len(free_seats_by_round[round_idx]) < 2:
                    continue
                seat1, seat2 = random.sample(free_seats_by_round[round_idx], 2)
                person1, person2 = arrangement[seat1], arrangement[seat2]
                
                # Skip if either person is locked
                if person1 in locked_people or person2 in locked_people:
                    continue
                
                # Swap them
                arrangement[seat1], arrangement[seat2] = person2, person1
            else:
                # Choose a block width randomly from the available range
                k = random.randint(1, max_width)
                
                # Identify the block of seats to flip
                if corner == "left":
                    block_seats = [(table_id, row, col) for row in [0, 1] for col in range(k)]
                else:  # right corner
                    block_seats = [(table_id, row, col) for row in [0, 1] for col in range(seats_per_side - k, seats_per_side)]
                
                # Filter out locked seats
                block_seats = [seat for seat in block_seats if seat not in locked_seats]
                
                # Create a new assignment by flipping each row's block
                for row in [0, 1]:
                    if corner == "left":
                        row_seats = [(table_id, row, col) for col in range(k) if (table_id, row, col) not in locked_seats]
                    else:
                        row_seats = [(table_id, row, col) for col in range(seats_per_side - k, seats_per_side) if (table_id, row, col) not in locked_seats]
                    
                    # Only include seats that are in the arrangement and not locked
                    row_seats = [seat for seat in row_seats if seat in arrangement and seat not in locked_seats]
                    if not row_seats:
                        continue
                    
                    # Get the current people in these seats
                    people = [arrangement[seat] for seat in row_seats]
                    
                    # Skip if any of these people are locked
                    if any(person in locked_people for person in people):
                        continue
                    
                    # Reverse the people and reassign
                    reversed_people = list(reversed(people))
                    for seat, person in zip(row_seats, reversed_people):
                        arrangement[seat] = person
        
        elif move_type_rand < flip_probability + adjacent_swap_probability:
            # ------------------ ADJACENT SWAP MOVE ------------------
            # Find all pairs of adjacent seats where both are free
            adjacent_free_pairs = []
            
            for seat in free_seats_by_round[round_idx]:
                if seat in arrangement:
                    person = arrangement[seat]
                    # Skip if this person is locked
                    if person in locked_people:
                        continue
                        
                    # Check side neighbors
                    for neighbor_seat in seat_neighbors[seat]["side"]:
                        if (neighbor_seat in free_seats_by_round[round_idx] and 
                            neighbor_seat in arrangement):
                            neighbor = arrangement[neighbor_seat]
                            # Skip if neighbor is locked
                            if neighbor in locked_people:
                                continue
                            # Add this pair to our list
                            adjacent_free_pairs.append((seat, neighbor_seat))
            
            # If no adjacent free pairs, fall back to random swap
            if not adjacent_free_pairs:
                if len(free_seats_by_round[round_idx]) < 2:
                    continue
                seat1, seat2 = random.sample(free_seats_by_round[round_idx], 2)
                person1, person2 = arrangement[seat1], arrangement[seat2]
                
                # Skip if either person is locked
                if person1 in locked_people or person2 in locked_people:
                    continue
                
                # Swap them
                arrangement[seat1], arrangement[seat2] = person2, person1
            else:
                # Choose a random adjacent pair
                seat1, seat2 = random.choice(adjacent_free_pairs)
                person1, person2 = arrangement[seat1], arrangement[seat2]
                
                # Swap them
                arrangement[seat1], arrangement[seat2] = person2, person1
        
        else:
            # ------------------ RANDOM SWAP MOVE ------------------
            # Randomly select two free seats to swap
            free_seats = free_seats_by_round[round_idx]
            if len(free_seats) < 2:
                continue
                
            seat1, seat2 = random.sample(free_seats, 2)
            person1, person2 = arrangement[seat1], arrangement[seat2]
            
            # Skip if either person is locked
            if person1 in locked_people or person2 in locked_people:
                continue
            
            # Swap them
            arrangement[seat1], arrangement[seat2] = person2, person1
        
        # Make sure to update the working_arrangements with the modified arrangement
        working_arrangements[round_idx] = arrangement
        
        # Calculate new score
        new_score_result = score_all_arrangements(arrangements_to_score=working_arrangements, track_guest_costs=False)
        if isinstance(new_score_result, tuple):
            new_score, _ = new_score_result
        else:
            new_score = new_score_result
        
        # Decide whether to accept the move
        accept_move = False
        if new_score > current_score:
            # Accept any improvement to current solution
            current_score = new_score
            current_arrangements = copy.deepcopy(working_arrangements)
            accept_move = True
            
            # Update best solution if this is better
            if new_score > best_score:
                best_score = new_score
                best_arrangements = copy.deepcopy(working_arrangements)
        elif random.random() < math.exp((new_score - current_score) / temperature):
            # Sometimes accept worse solutions based on temperature
            current_score = new_score
            current_arrangements = copy.deepcopy(working_arrangements)
            accept_move = True
        
        # Record progress periodically
        if iteration % update_freq == 0 or iteration == max_iterations - 1:
            # Always record the BEST score, not the current one
            iterations.append(iteration + 1)
            best_scores.append(best_score)
            
            # Get component breakdown for BEST arrangements
            _, components = score_all_arrangements(collect_components=True, arrangements_to_score=best_arrangements)
            
            for component, value in components.items():
                component_scores[component].append(value)
            
            # Update the progress chart
            progress_df = pd.DataFrame({
                'Iteration': iterations,
                'Best Score': best_scores
            })
            
            # Create a line chart for the best score
            progress_chart.line_chart(progress_df.set_index('Iteration'))
        
        # Cool down
        temperature *= cooling_rate
    
    # Final detailed breakdown
    with cost_breakdown:
        st.write("### Final Cost Breakdown")
        
        # Create a DataFrame for component scores
        component_df = pd.DataFrame({
            'Iteration': iterations,
            **{component: values for component, values in component_scores.items()}
        })
        
        # Display the component chart
        st.line_chart(component_df.set_index('Iteration'))
        
        # Display the final component values
        st.write("### Final Component Values")
        final_components = {component: values[-1] for component, values in component_scores.items()}
        final_components['Total'] = sum(final_components.values())
        st.json(final_components)
    
    # Calculate final guest costs for the best arrangement
    _, _, final_guest_costs = score_all_arrangements(collect_components=True, arrangements_to_score=best_arrangements, track_guest_costs=True)
    
    # Display final problematic guests
    with problematic_guests:
        st.write("### Guest Cost Breakdown")
        display_problematic_guests(final_guest_costs)
    
    return best_arrangements

def display_problematic_guests(guest_costs):
    """Display the guests with the highest costs, broken down by cost type"""
    if not guest_costs:
        st.write("No guest cost data available")
        return
    
    # Convert to DataFrame for easier manipulation
    rows = []
    for person, costs in guest_costs.items():
        row = {'Guest': person}
        row.update(costs)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by total cost (most problematic first)
    if 'total' in df.columns:
        df = df.sort_values('total')
    
    # Display the table
    st.dataframe(df)
    
    # Show the top 5 most problematic guests
    if len(df) > 0 and 'total' in df.columns:
        worst_guests = df.sort_values('total').head(5)
        st.write("### Top 5 Most Problematic Guests")
        
        # Create a bar chart of the worst guests
        st.bar_chart(worst_guests.set_index('Guest')['total'])

def set_fixed_assignments():
    with st.sidebar.expander("Fixed Assignments"):
        st.markdown("""
            Specify fixed seat assignments for specific rounds.
            Format: Round#:SeatID:Name (e.g., 1:A1:Johan)
        """)
        
        settings = load_settings()
        fixed_assignments_text = st.text_area(
            "Fixed Assignments", 
            value=get_setting(settings, Settings.FIXED_ASSIGNMENTS),
            height=150,
            key=str(Settings.FIXED_ASSIGNMENTS),
            help="Format: Round#:SeatID:Name (e.g., 1:A1:Johan)"
        )
        
        # Parse fixed assignments
        locked_seats_per_round = defaultdict(dict)
        for line in fixed_assignments_text.strip().splitlines():
            if not line.strip():
                continue
            try:
                parts = line.split(":")
                if len(parts) == 3:
                    round_num, seat_id, name = parts
                    round_idx = int(round_num.strip()) - 1  # Convert to 0-based index
                    seat_id = seat_id.strip().upper()
                    name = name.strip()
                    # Check if this seat is already assigned for this round
                    if seat_id in locked_seats_per_round[round_idx]:
                        st.warning(f"Seat {seat_id} in round {round_num} is assigned multiple times. Last assignment will be used.")
                    locked_seats_per_round[round_idx][seat_id] = name
            except Exception:
                st.error(f"Error parsing line: {line}")
        
        st.session_state.locked_seats_per_round = dict(locked_seats_per_round)

def import_export_settings():
    """Handle import and export of settings"""
    st.sidebar.markdown("## Import/Export")
    
    # Import settings
    uploaded_file = st.sidebar.file_uploader("Import Settings", type=['json'])
    if uploaded_file is not None:
        try:
            settings = json.load(uploaded_file)
            st.session_state.uploaded_settings = settings
            st.sidebar.success("Settings loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading settings: {str(e)}")
    
    # Export settings toggle
    if "show_export" not in st.session_state:
        st.session_state.show_export = False

    if st.sidebar.button("Export Settings"):
        st.session_state.show_export = not st.session_state.show_export

    # Show export section if toggled
    if st.session_state.show_export:
        # Collect all current settings
        current_settings = {}
        for setting in Settings:
            setting_key = str(setting)
            if setting_key in st.session_state:
                current_settings[setting_key] = st.session_state[setting_key]
        
        # Convert to JSON string with proper formatting
        settings_json = json.dumps(current_settings, indent=2)
        
        st.sidebar.download_button(
            label="👉 Click here to download Settings JSON",
            data=settings_json,
            file_name="seatplan_settings.json",
            mime="application/json",
            key="settings_download"
        )
        st.sidebar.write("Settings preview:")
        st.sidebar.json(current_settings)

def main():
    st.set_page_config(layout="wide")

    st.title("SeatPlan v2")
    
    import_export_settings()
    
    set_tables()
    set_guests()
    set_fixed_assignments()
    
    TABLES, TABLE_LETTERS = parse_table_definitions(st.session_state[str(Settings.TABLE_DEFINITIONS)])

    guests = st.session_state.guests
    
    st.markdown("## Seating Generation")
    cols = st.columns([1, 1, 5])  
    with cols[0]:
        num_rounds = st.number_input("Number of arrangements", min_value=1, max_value=5, value=3, step=1)
        use_optimization = st.checkbox("Use optimization", value=True)
        if use_optimization:
            max_iterations = st.number_input("Optimization iterations", min_value=100, max_value=100000, value=1000, step=100)
        else:
            max_iterations = 0

    # Check if we have previous arrangements to build upon
    previous_arrangements = None
    if "previous_arrangements" in st.session_state:
        use_previous = st.checkbox("Build upon previous arrangements", value=False)
        if use_previous:
            previous_arrangements = st.session_state.previous_arrangements
    
    # Add a prominent run button to trigger the optimization
    st.markdown("### Generate New Arrangements")
    run_button = st.button("Generate Seating Arrangements", key="generate_button", use_container_width=True)
    
    if run_button or "arrangements" not in st.session_state:
        # Get gender information
        male_names = parse_lines(st.session_state[str(Settings.MALES)])
        female_names = parse_lines(st.session_state[str(Settings.FEMALES)])
        person_genders = {}
        for name in male_names:
            person_genders[name] = "M"
        for name in female_names:
            person_genders[name] = "F"
        
        # Get preferences
        preferred_neighbors = st.session_state.preferred_neighbors if hasattr(st.session_state, 'preferred_neighbors') else {}
        excluded_neighbors = st.session_state.excluded_neighbors if hasattr(st.session_state, 'excluded_neighbors') else {}
        
        # Generate arrangements
        locked_seats_per_round = st.session_state.locked_seats_per_round if hasattr(st.session_state, 'locked_seats_per_round') else {}
        
        arrangements = run_optimization(
            TABLES, 
            guests, 
            num_rounds, 
            locked_seats_per_round, 
            table_letters=TABLE_LETTERS,
            previous_arrangements=previous_arrangements,
            person_genders=person_genders if use_optimization else None,
            preferred_neighbors=preferred_neighbors if use_optimization else None,
            excluded_neighbors=excluded_neighbors if use_optimization else None,
            max_iterations=max_iterations
        )
        
        # Store arrangements for potential future use
        st.session_state.previous_arrangements = arrangements
        st.session_state.arrangements = arrangements
    
    # Display the current arrangements
    if "arrangements" in st.session_state:
        show_arrangements(st.session_state.arrangements, TABLES, TABLE_LETTERS)

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
