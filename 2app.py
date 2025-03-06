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

_FROM_INPUT = "_from_input"

class Weights(Enum):
    PREFERRED_NEIGHBOR = "preferred_neighbor"
    EXCLUDED_NEIGHBOR = "excluded_neighbor"
    GENDER_ALTERNATION = "gender_alternation"
    REPEAT_NEIGHBOR = "repeat_neighbor"
    CORNER = "corner"
    
    def __str__(self):
        return self.value

class OptimizationParams(Enum):
    ITERATIONS = "iterations"
    INITIAL_TEMP = "initial_temp"
    COOLING_RATE = "cooling_rate"
    NUM_ROUNDS = "num_rounds"

    def __str__(self):
        return self.value

class Settings(Enum):
    TABLE_DEFINITIONS = "table_definitions"
    MALES = "males"
    FEMALES = "females"
    FIXED_ASSIGNMENTS = "fixed_assignments"
    PREFERRED_NEIGHBORS = "preferred_neighbors"
    EXCLUDED_NEIGHBORS = "excluded_neighbors"
    SAME_GENDER_OK = "same_gender_ok"
    REPEAT_OK_GROUPS = "repeat_ok_groups"
    WEIGHTS = "weights"
    OPTIMIZATION_PARAMS = "optimization_params"
    def __str__(self):
        return self.value

DEFAULT_SETTINGS = json.load(open("default_settings.json", "r", encoding="utf-8"))


##### 

def compute_seat_neighbors(tables):
    """
    Compute neighbors for each seat in the tables.
    Returns a dictionary mapping each seat to its neighbors.
    """
    neighbors = {}
    for table_id, seats_per_side in tables.items():
        for row in [0, 1]:
            for col in range(seats_per_side):
                seat = (table_id, row, col)
                side_neighbors = []
                front_neighbors = []
                diagonal_neighbors = []
                
                # Side neighbors (left and right)
                if col > 0:
                    side_neighbors.append((table_id, row, col - 1))  # Left
                if col < seats_per_side - 1:
                    side_neighbors.append((table_id, row, col + 1))  # Right
                
                # Front neighbors (across the table)
                front_neighbors.append((table_id, 1 - row, col))  # Directly across
                
                # Diagonal neighbors (across the table, diagonally)
                if col > 0:
                    diagonal_neighbors.append((table_id, 1 - row, col - 1))  # Diagonal left
                if col < seats_per_side - 1:
                    diagonal_neighbors.append((table_id, 1 - row, col + 1))  # Diagonal right
                
                # Combine all neighbors
                all_neighbors = side_neighbors + front_neighbors + diagonal_neighbors
                
                neighbors[seat] = {
                    "side": side_neighbors,
                    "front": front_neighbors,
                    "diagonal": diagonal_neighbors,
                    "all": all_neighbors
                }
    return neighbors


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

def generate_arrangement_html(arrangement, table_id, tables, locked_seats_per_round=None, round_idx=None, highlight_guest=None, all_arrangements=None, show_locked_seats=True, show_gender=False):
    num_cols = tables[table_id]
    cell_style_base = (
        "width:60px; height:60px; border:1px solid #000; display:flex; "
        "align-items:center; justify-content:center; margin:1px; font-size:12px; font-weight:bold;"
        "text-align:center; word-break:break-word; overflow:hidden;"
    )
    
    # Get seat neighbors for highlighting
    seat_neighbors = None
    highlight_seat = None
    
    # Track all neighbors of the highlighted guest across all rounds by type
    all_side_neighbors = set()
    all_front_neighbors = set()
    all_diagonal_neighbors = set()
    
    if highlight_guest and all_arrangements:
        seat_neighbors = compute_seat_neighbors(tables)
        
        # Find the seat of the highlighted guest in current round
        for seat, guest in arrangement.items():
            if guest == highlight_guest and seat[0] == table_id:
                highlight_seat = seat
                break
        
        # Collect all neighbors across all rounds (including current)
        for r_arrangement in all_arrangements:
            # Find the guest's seat in this round
            guest_seat = None
            for seat, guest in r_arrangement.items():
                if guest == highlight_guest:
                    guest_seat = seat
                    break
            
            if guest_seat:
                # Add side neighbors
                for neighbor_seat in seat_neighbors[guest_seat]["side"]:
                    if neighbor_seat in r_arrangement:
                        all_side_neighbors.add(r_arrangement[neighbor_seat])
                
                # Add front neighbors
                for neighbor_seat in seat_neighbors[guest_seat]["front"]:
                    if neighbor_seat in r_arrangement:
                        all_front_neighbors.add(r_arrangement[neighbor_seat])
                
                # Add diagonal neighbors
                for diagonal_seat in seat_neighbors[guest_seat]["diagonal"]:
                    if diagonal_seat in r_arrangement:
                        all_diagonal_neighbors.add(r_arrangement[diagonal_seat])
    
    # Get locked seats for this round
    locked_seats = set()
    if show_locked_seats and locked_seats_per_round and round_idx is not None and round_idx in locked_seats_per_round:
        for seat_label, _ in locked_seats_per_round[round_idx].items():
            seat_tuple = seat_label_to_tuple(seat_label, tables, {table_id: chr(65 + table_id)})
            if seat_tuple and seat_tuple[0] == table_id:
                locked_seats.add(seat_tuple)
    
    def get_cell_style(seat, occupant):
        style = cell_style_base
        bg_color = "#ffffff"  # Default white background
        border = "1px solid #000"  # Default border
        text_color = "#000000"  # Default black text
        
        # Show gender colors if enabled
        if show_gender and occupant and occupant in st.session_state.person_genders:
            if st.session_state.person_genders[occupant] == "M":
                bg_color = "#add8e6"  # Light blue for males
            else:
                bg_color = "#e6b3e6"  # Light purple for females
        
        # Highlight fixed seats with red border
        if seat in locked_seats:
            border = "2px solid #ff0000"  # Red border for fixed seats
        
        # Highlight the selected guest and their neighbors with priority
        if highlight_guest:
            if occupant == highlight_guest:
                bg_color = "#ffcc00"  # Yellow for selected guest
            # Priority 1: Side neighbors (highest interaction)
            elif occupant in all_side_neighbors:
                bg_color = "#99ff99"  # Light green for side neighbors
            # Priority 2: Front neighbors (across table)
            elif occupant in all_front_neighbors:
                bg_color = "#ccffcc"  # Lighter green for front neighbors
            # Priority 3: Diagonal neighbors (lowest interaction)
            elif occupant in all_diagonal_neighbors:
                bg_color = "#add8e6"  # Light blue for diagonal neighbors
        
        return f"{style} background-color:{bg_color}; border:{border};"
    
    # Use flex with no wrapping and enable horizontal scrolling
    container_style = "display:flex; justify-content:center; flex-wrap:nowrap; overflow-x:auto;"
    
    top_html = f"<div style='{container_style}'>"
    for col in range(num_cols):
        seat = (table_id, 0, col)
        occupant = arrangement.get(seat, "")
        cell_style = get_cell_style(seat, occupant)
        top_html += f"<div style='{cell_style}'>{occupant}</div>"
    top_html += "</div>"
    
    bottom_html = f"<div style='{container_style}'>"
    for col in range(num_cols):
        seat = (table_id, 1, col)
        occupant = arrangement.get(seat, "")
        cell_style = get_cell_style(seat, occupant)
        bottom_html += f"<div style='{cell_style}'>{occupant}</div>"
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

def set_same_gender_ok():
    with st.sidebar.expander("Gender Neutral"):
        settings = load_settings()
        same_gender_ok_input = st.text_area("These are not penalized for sitting with the same gender", value=get_setting(settings, Settings.SAME_GENDER_OK), height=150, key=str(Settings.SAME_GENDER_OK))
        
        same_gender_ok = same_gender_ok_input.split(",")
        st.session_state[str(Settings.SAME_GENDER_OK)+_FROM_INPUT] = same_gender_ok
        
        st.caption(f"Number of people who are okay with the same gender: {len(same_gender_ok)}")

def set_groups():
    with st.sidebar.expander("Groups"):
        st.markdown("""
            Each line represents a group of guests who are okay to sit together repeatedly. 
            Names are separated with commas.
        """)
        settings = load_settings()
        groups_input = st.text_area("Groups", value=get_setting(settings, Settings.REPEAT_OK_GROUPS), height=150, key=str(Settings.REPEAT_OK_GROUPS))
        
        groups = []
        for line in groups_input.splitlines():
            groups.append(line.strip().split(","))
        st.session_state[str(Settings.REPEAT_OK_GROUPS)+_FROM_INPUT] = groups
        st.caption(f"Number of groups: {len(st.session_state.repeat_ok_groups_from_input)}")
        st.caption(f"People in groups: {len([person for group in st.session_state.repeat_ok_groups_from_input for person in group])}")

def parse_neighbor_relationships(input_text):
    """
    Parse neighbor relationships from text input.
    Format: Name: Neighbor1, Neighbor2, ...
    Returns a dictionary mapping names to lists of neighbors.
    """
    relationships = {}
    name_occurrences = {}  # Track how many times each name appears
    
    for line in input_text.splitlines():
        if not line.strip():
            continue
        try:
            parts = line.split(":")
            if len(parts) == 2:
                name = parts[0].strip()
                name_occurrences[name] = name_occurrences.get(name, 0) + 1
                neighbors = [x.strip() for x in parts[1].split(",") if x.strip()]
                relationships[name] = neighbors
        except Exception as e:
            st.error(f"Error parsing line: {line}")
    
    # Alert about duplicate entries
    duplicate_names = [name for name, count in name_occurrences.items() if count > 1]
    if duplicate_names:
        st.warning(f"You have duplicate entries for: {', '.join(duplicate_names)}")
    
    return relationships

def set_preferred_neighbors():
    with st.sidebar.expander("Preferred Neighbors"):
        settings = load_settings()
        preferred_neighbors_input = st.text_area("Preferred Neighbors", 
                                               value=get_setting(settings, Settings.PREFERRED_NEIGHBORS), 
                                               height=150, 
                                               key=str(Settings.PREFERRED_NEIGHBORS))
        
        preferred_neighbors = parse_neighbor_relationships(preferred_neighbors_input)
        st.session_state[str(Settings.PREFERRED_NEIGHBORS)+_FROM_INPUT] = preferred_neighbors
        
        st.caption(f"Number of people with preferences: {len(preferred_neighbors)}")

def set_excluded_neighbors():
    with st.sidebar.expander("Excluded Neighbors"):
        settings = load_settings()
        excluded_neighbors_input = st.text_area("Excluded Neighbors", 
                                              value=get_setting(settings, Settings.EXCLUDED_NEIGHBORS), 
                                              height=150, 
                                              key=str(Settings.EXCLUDED_NEIGHBORS))
        
        excluded_neighbors = parse_neighbor_relationships(excluded_neighbors_input)
        st.session_state[str(Settings.EXCLUDED_NEIGHBORS)+_FROM_INPUT] = excluded_neighbors
        
        st.caption(f"Number of people with exclusions: {len(excluded_neighbors)}")

def set_guests():
    with st.sidebar.expander("Guests"):
        st.markdown("""
            Enter the names of the guests.
        """)
        
        settings = load_settings()
        
        male_text = st.text_area("Males (one per line)", 
                                  value=get_setting(settings, Settings.MALES), 
                                  height=300,
                                  key=str(Settings.MALES), 
                                  help="One male name per line.")
        male_count = len([name for name in male_text.splitlines() if name.strip()])
        st.caption(f"Male count: {male_count}")
        
        female_text = st.text_area("Females (one per line)", 
                                   value=get_setting(settings, Settings.FEMALES), 
                                   height=300,
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

        st.session_state.person_genders = person_genders
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
    cols = st.columns([1, 1, 1, 1, 3])  
    with cols[0]:
        tables_per_row = st.number_input("Tables per row", min_value=1, max_value=5, value=3, step=1)
    
    with cols[1]:
        highlight_guest = st.selectbox("Highlight guest", ["None"] + sorted(list(set([guest for arr in arrangements for guest in arr.values()]))))
        highlight_guest = None if highlight_guest == "None" else highlight_guest
    
    with cols[2]:
        show_locked_seats = st.checkbox("Show locked seats", value=True)
        
    with cols[3]:
        show_gender = st.checkbox("Show gender", value=False)
    
    # Display legend once at the top if a guest is highlighted
    if highlight_guest:
        st.markdown("""
        <div style="margin-bottom:15px;">
          <span style="background-color:#ffcc00; padding:2px 5px; margin-right:10px; color:black;">Selected Guest</span>
          <span style="background-color:#99ff99; padding:2px 5px; margin-right:10px; color:black;">Side Neighbors</span>
          <span style="background-color:#ccffcc; padding:2px 5px; margin-right:10px; color:black;">Front Neighbors</span>
          <span style="background-color:#add8e6; padding:2px 5px; color:black;">Diagonal Neighbors</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Display legend for gender if enabled
    if show_gender:
        st.markdown("""
        <div style="margin-bottom:15px;">
          <span style="background-color:#add8e6; padding:2px 5px; margin-right:10px;">Male</span>
          <span style="background-color:#e6b3e6; padding:2px 5px;">Female</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Add legend for locked seats if they're being shown
    if show_locked_seats:
        st.markdown("""
        <div style="margin-bottom:15px;">
          <span style="background-color:#ffffff; border:2px solid #ff0000; padding:2px 5px; color:black;">Locked Seat</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Display each arrangement
    for round_idx, arrangement in enumerate(arrangements):
        st.markdown(f"### Arrangement {round_idx + 1}")
        
        # Display each table
        table_ids = sorted(tables.keys())
        
        for i in range(0, len(table_ids), tables_per_row):
            cols = st.columns(tables_per_row)
            for j, table_id in enumerate(table_ids[i:i+tables_per_row]):
                if i + j < len(table_ids):
                    with cols[j]:
                        table_letter = table_letters[table_id]
                        st.markdown(f"**Table {table_letter}**")
                        html = generate_arrangement_html(
                            arrangement, 
                            table_id, 
                            tables, 
                            locked_seats_per_round=st.session_state.get(str(Settings.FIXED_ASSIGNMENTS)+_FROM_INPUT),
                            round_idx=round_idx,
                            highlight_guest=highlight_guest,
                            all_arrangements=arrangements,
                            show_locked_seats=show_locked_seats,
                            show_gender=show_gender,
                        )
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

def run_optimization(tables, guests, num_rounds, locked_seats_per_round, table_letters=None, 
                    previous_arrangements=None, person_genders=None, preferred_neighbors=None, 
                    excluded_neighbors=None, same_gender_ok=None, repeat_ok_groups=None, max_iterations=5000):
    """
    Run the optimization algorithm to generate seating arrangements.
    
    Args:
        tables: Dictionary mapping table IDs to number of seats per side
        guests: List of guest names
        num_rounds: Number of seating arrangements to generate
        locked_seats_per_round: Dictionary mapping rounds to locked seat assignments
        table_letters: Dictionary mapping table IDs to table letters
        previous_arrangements: Optional list of previous arrangements to build upon
        person_genders: Optional dictionary mapping person names to genders
        preferred_neighbors: Optional dictionary mapping person names to lists of preferred neighbors
        excluded_neighbors: Optional dictionary mapping person names to lists of excluded neighbors
        same_gender_ok: Optional list of people who are okay sitting with the same gender
        repeat_ok_groups: Optional list of groups who are okay sitting together repeatedly
        max_iterations: Maximum number of iterations for optimization
        
    Returns:
        List of seating arrangements
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
        available_males = [g for g in available_guests if person_genders.get(g) == "M"]
        available_females = [g for g in available_guests if person_genders.get(g) == "F"]
        
        # For initial arrangement, separate by gender for alternating pattern
        if person_genders:
            
            # Initialize pattern starting with female in upper left
            for table_id in tables:
                for row in [0, 1]:
                    for col in range(tables[table_id]):
                        seat = (table_id, row, col)
                        if seat not in arrangement:
                            # Determine parity of position (female in upper left, then alternate)
                            # For even parity positions (both odd/both even indices), assign female
                            is_female_position = ((row + col) % 2 == 0)
                            
                            if is_female_position:
                                # Try to assign a female first, but if none available, try a male
                                if available_females:
                                    person = available_females.pop(0)
                                elif available_males:  # If no females left, use males
                                    person = available_males.pop(0)
                                elif available_guests:  # Fallback to any remaining guest
                                    person = available_guests.pop(0)
                                else:
                                    continue  # Skip if no guests available
                                arrangement[seat] = person
                            else:
                                # Try to assign a male first, but if none available, try a female
                                if available_males:
                                    person = available_males.pop(0)
                                elif available_females:  # If no males left, use females
                                    person = available_females.pop(0)
                                elif available_guests:  # Fallback to any remaining guest
                                    person = available_guests.pop(0)
                                else:
                                    continue  # Skip if no guests available
                                arrangement[seat] = person
        
        arrangements.append(arrangement)
    
    # If we're using optimization, run the optimizer
    if max_iterations > 0:
        arrangements = optimize_all_arrangements(
            arrangements, seats, tables, table_letters, seat_neighbors, 
            person_genders, preferred_neighbors, excluded_neighbors, 
            locked_seats_per_round, same_gender_ok, repeat_ok_groups, max_iterations
        )
    
    return arrangements

def optimize_all_arrangements(arrangements, seats, tables, table_letters, seat_neighbors, person_genders, 
                             preferred_neighbors, excluded_neighbors, locked_seats_per_round,
                             same_gender_ok, repeat_ok_groups, max_iterations):
    """
    Optimize all seating arrangements together, considering:
    1. Alternating male/female within each arrangement
    2. Preferred neighbors within each arrangement
    3. Excluded neighbors within each arrangement
    4. Avoiding repeat neighbors across different arrangements
    
    Uses a local search algorithm with random swaps and flips.
    """
    # Get weights from session state or use defaults
    weights = st.session_state.get(str(Settings.WEIGHTS), {})
    
    # Weights for different optimization criteria
    GENDER_ALTERNATION_WEIGHT = weights.get(str(Weights.GENDER_ALTERNATION), 150)  # Weight for gender alternation
    PREFERRED_NEIGHBOR_WEIGHT = weights.get(str(Weights.PREFERRED_NEIGHBOR), 20)  # Weight for preferred neighbors
    EXCLUDED_NEIGHBOR_WEIGHT = weights.get(str(Weights.EXCLUDED_NEIGHBOR), 50)   # Weight for excluded neighbors
    REPEAT_NEIGHBOR_WEIGHT = weights.get(str(Weights.REPEAT_NEIGHBOR), 10)     # Weight for repeat neighbors across rounds
    CORNER_PENALTY_WEIGHT = weights.get(str(Weights.CORNER), 10)  # Weight for corner position penalty

    # st.write(f"GENDER_ALTERNATION_WEIGHT: {GENDER_ALTERNATION_WEIGHT}")
    # st.write(f"PREFERRED_NEIGHBOR_WEIGHT: {PREFERRED_NEIGHBOR_WEIGHT}")
    # st.write(f"EXCLUDED_NEIGHBOR_WEIGHT: {EXCLUDED_NEIGHBOR_WEIGHT}")
    # st.write(f"REPEAT_NEIGHBOR_WEIGHT: {REPEAT_NEIGHBOR_WEIGHT}")
    # st.write(f"CORNER_PENALTY_WEIGHT: {CORNER_PENALTY_WEIGHT}")

    
    # Ensure same_gender_ok is a list
    if same_gender_ok is None:
        same_gender_ok = []
    
    # Ensure repeat_ok_groups is a list
    if repeat_ok_groups is None:
        repeat_ok_groups = []
    
    # Create a lookup dictionary for repeat_ok_groups
    # For each person, store which group they belong to
    person_to_group = {}
    for group_idx, group in enumerate(repeat_ok_groups):
        for person in group:
            person_to_group[person] = group_idx
    
    # Create reverse mappings from person to seat for each arrangement
    person_to_seat_by_round = []
    for arrangement in arrangements:
        person_to_seat = {person: seat for seat, person in arrangement.items()}
        person_to_seat_by_round.append(person_to_seat)
    
    # Create a placeholder for the progress chart
    progress_chart = st.empty()
    
    # Create containers for detailed cost breakdown
    cost_breakdown = st.expander("Cost Function Breakdown", expanded=False)
    
    # Create container for problematic guests
    problematic_guests = st.expander("Guest Costs", expanded=False)
    
    # Lists to track progress
    iterations = []
    best_scores = []  # Only track best scores
    component_scores = {
        "Gender Alternation": [],
        "Preferred Neighbors": [],
        "Excluded Neighbors": [],
        "Repeat Neighbors": [],
        "Corner Positions": []
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
        corner_score = 0
        
        # Track costs per guest if requested
        guest_costs = defaultdict(lambda: defaultdict(int)) if track_guest_costs else None
        
        # Track all neighbor pairs across all rounds
        all_neighbor_pairs = set()
        
        # Track who sits in corner positions in each round
        corner_positions_by_person = defaultdict(list)
        
        # Score each arrangement individually
        for round_idx, arrangement in enumerate(arrangements_to_score):
            round_score = 0
            
            # Identify corner positions for this round
            corner_positions = []
            for table_id, seats_per_side in tables.items():
                # Left corners (first column)
                corner_positions.append((table_id, 0, 0))  # Top-left
                corner_positions.append((table_id, 1, 0))  # Bottom-left
                
                # Right corners (last column)
                corner_positions.append((table_id, 0, seats_per_side - 1))  # Top-right
                corner_positions.append((table_id, 1, seats_per_side - 1))  # Bottom-right
            
            # Track who sits in corner positions
            for corner in corner_positions:
                if corner in arrangement:
                    person = arrangement[corner]
                    corner_positions_by_person[person].append(round_idx)
            
            # Score gender alternation
            if person_genders:
                for seat in seats:
                    if seat in arrangement:
                        person = arrangement[seat]
                        person_gender = person_genders.get(person)
                        
                        # Skip gender alternation check if this person is okay with same gender
                        if person in same_gender_ok:
                            continue
                        
                        # Check side neighbors (left/right)
                        for neighbor_seat in seat_neighbors[seat]["side"]:
                            if neighbor_seat in arrangement:
                                neighbor = arrangement[neighbor_seat]
                                neighbor_gender = person_genders.get(neighbor)
                                
                                # Skip if neighbor is okay with same gender
                                if neighbor in same_gender_ok:
                                    continue
                                
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
                        
                        # Check front neighbors (across the table)
                        for neighbor_seat in seat_neighbors[seat]["front"]:
                            if neighbor_seat in arrangement:
                                neighbor = arrangement[neighbor_seat]
                                neighbor_gender = person_genders.get(neighbor)
                                
                                # Skip if neighbor is okay with same gender
                                if neighbor in same_gender_ok:
                                    continue
                                
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
                            for neighbor_seat in seat_neighbors[seat]["side"]:
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
                            for neighbor_seat in seat_neighbors[seat]["side"]:
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
                # Skip penalty if both people are in the same repeat-ok group
                person1, person2 = pair
                group1 = person_to_group.get(person1)
                group2 = person_to_group.get(person2)
                
                # If both people are in a group and it's the same group, skip the penalty
                if group1 is not None and group2 is not None and group1 == group2:
                    continue
                
                # Apply increasing penalty for each repeat
                repeat_penalty = -sum(range(len(rounds))) * REPEAT_NEIGHBOR_WEIGHT
        
        # Penalize repeat corner positions
        for person, rounds in corner_positions_by_person.items():
            if len(rounds) > 1:
                # Apply exponential penalty for repeat corner positions
                corner_penalty = -((len(rounds) - 1) ** 2) * CORNER_PENALTY_WEIGHT
                total_score += corner_penalty
                if collect_components:
                    corner_score += corner_penalty
                if track_guest_costs:
                    guest_costs[person]["corner"] += corner_penalty
        
        if track_guest_costs:
            # Calculate total cost per guest
            for person in guest_costs:
                guest_costs[person]["total"] = sum(guest_costs[person].values())
            
            if collect_components:
                return total_score, {
                    "Gender Alternation": gender_score,
                    "Preferred Neighbors": preferred_score,
                    "Excluded Neighbors": excluded_score,
                    "Repeat Neighbors": repeat_score,
                    "Corner Positions": corner_score
                }, guest_costs
            return total_score, guest_costs
        
        if collect_components:
            return total_score, {
                "Gender Alternation": gender_score,
                "Preferred Neighbors": preferred_score,
                "Excluded Neighbors": excluded_score,
                "Repeat Neighbors": repeat_score,
                "Corner Positions": corner_score
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
    
    # Function to get problematic guests and their seats
    def get_problematic_guests(current_guest_costs, round_idx):
        # Calculate total cost per guest
        total_costs = {person: costs["total"] for person, costs in current_guest_costs.items()}
        
        # Sort guests by cost (most problematic first)
        sorted_guests = sorted(total_costs.items(), key=lambda x: x[1])
        
        # Get the seats of the problematic guests in the current round
        problematic_seats = []
        for person, cost in sorted_guests:
            if person in person_to_seat_by_round[round_idx]:
                seat = person_to_seat_by_round[round_idx][person]
                problematic_seats.append((seat, cost))
        
        return problematic_seats
    
    # Function to sample seats with bias towards problematic guests
    def sample_seats_near_problematic(free_seats, problematic_seats, num_samples=2):
        if not problematic_seats or not free_seats:
            # Fall back to random sampling if no problematic seats
            return random.sample(free_seats, min(num_samples, len(free_seats)))
        
        # Calculate weights based on proximity to problematic guests
        weights = []
        for seat in free_seats:
            # Calculate minimum distance to any problematic guest
            min_distance = float('inf')
            for prob_seat, cost in problematic_seats:
                # Simple distance metric: same table = 1, different table = 2
                if seat[0] == prob_seat[0]:  # Same table
                    distance = 1
                else:
                    distance = 2
                
                # Adjust by cost (higher cost = lower distance)
                adjusted_distance = distance / (abs(cost) + 1)
                min_distance = min(min_distance, adjusted_distance)
            
            # Weight is inverse of distance (closer = higher weight)
            weight = 1.0 / (min_distance + 0.1)  # Add small constant to avoid division by zero
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(free_seats)] * len(free_seats)
        
        # Sample seats based on weights
        sampled_indices = random.choices(range(len(free_seats)), weights=weights, k=num_samples)
        return [free_seats[i] for i in sampled_indices]
    
    # Get current guest costs for targeting problematic guests
    current_guest_costs = guest_costs
    
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
        
        # Update person_to_seat mapping for this round
        person_to_seat_by_round[round_idx] = {person: seat for seat, person in arrangement.items()}
        
        # Get problematic guests and their seats for this round
        problematic_seats = get_problematic_guests(current_guest_costs, round_idx)
        
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
                free_seats = free_seats_by_round[round_idx]
                if len(free_seats) < 2:
                    continue
                
                # Sample seats with bias towards problematic guests
                seat1, seat2 = sample_seats_near_problematic(free_seats, problematic_seats, 2)
                
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
            
            # Prioritize seats near problematic guests
            problematic_adjacent_pairs = []
            
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
                            
                            # Check if either person is problematic
                            is_problematic = False
                            for prob_seat, _ in problematic_seats:
                                if seat == prob_seat or neighbor_seat == prob_seat:
                                    is_problematic = True
                                    break
                            
                            if is_problematic:
                                problematic_adjacent_pairs.append((seat, neighbor_seat))
            
            # If we have problematic adjacent pairs, prioritize those
            if problematic_adjacent_pairs:
                seat1, seat2 = random.choice(problematic_adjacent_pairs)
                person1, person2 = arrangement[seat1], arrangement[seat2]
                
                # Swap them
                arrangement[seat1], arrangement[seat2] = person2, person1
            # If no adjacent free pairs, fall back to random swap
            elif not adjacent_free_pairs:
                free_seats = free_seats_by_round[round_idx]
                if len(free_seats) < 2:
                    continue
                
                # Sample seats with bias towards problematic guests
                seat1, seat2 = sample_seats_near_problematic(free_seats, problematic_seats, 2)
                
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
            # Randomly select two free seats to swap, with bias towards problematic guests
            free_seats = free_seats_by_round[round_idx]
            if len(free_seats) < 2:
                continue
                
            # Sample seats with bias towards problematic guests
            seat1, seat2 = sample_seats_near_problematic(free_seats, problematic_seats, 2)
            
            person1, person2 = arrangement[seat1], arrangement[seat2]
            
            # Skip if either person is locked
            if person1 in locked_people or person2 in locked_people:
                continue
            
            # Swap them
            arrangement[seat1], arrangement[seat2] = person2, person1
        
        # Make sure to update the working_arrangements with the modified arrangement
        working_arrangements[round_idx] = arrangement
        
        # Calculate new score
        new_score_result = score_all_arrangements(arrangements_to_score=working_arrangements, track_guest_costs=True)
        if isinstance(new_score_result, tuple):
            new_score, new_guest_costs = new_score_result
        else:
            new_score = new_score_result
            new_guest_costs = None
        
        # Decide whether to accept the move
        accept_move = False
        if new_score > current_score:
            # Accept any improvement to current solution
            current_score = new_score
            current_arrangements = copy.deepcopy(working_arrangements)
            if new_guest_costs:
                current_guest_costs = new_guest_costs
            accept_move = True
            
            # Update best solution if this is better
            if new_score > best_score:
                best_score = new_score
                best_arrangements = copy.deepcopy(working_arrangements)
                if new_guest_costs:
                    best_guest_costs = new_guest_costs
        elif random.random() < math.exp((new_score - current_score) / temperature):
            # Sometimes accept worse solutions based on temperature
            current_score = new_score
            current_arrangements = copy.deepcopy(working_arrangements)
            if new_guest_costs:
                current_guest_costs = new_guest_costs
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
    
    # Ensure all cost columns exist
    for col in ['gender', 'preferred', 'excluded', 'repeat', 'corner', 'total']:
        if col not in df.columns:
            df[col] = 0
    
    # Sort by total cost (most problematic first)
    df = df.sort_values('total', ascending=True)
    
    # Display the table
    st.dataframe(df)
    
    # Show the top 5 most problematic guests
    if len(df) > 0:
        worst_guests = df.sort_values('total').head(5)
        st.write("### Top 5 Most Problematic Guests")
        
        # Create a bar chart of the worst guests
        st.bar_chart(worst_guests.set_index('Guest')['total'])

def set_fixed_assignments():
    with st.sidebar.expander("Locked Seats"):
        settings = load_settings()
        fixed_assignments_text = st.text_area(
            "Fixed Assignments", 
            value=get_setting(settings, Settings.FIXED_ASSIGNMENTS),
            height=250,
            key=str(Settings.FIXED_ASSIGNMENTS),
            help="Format: Round#:SeatID:Name (e.g., 1:A1:Johan)"
        )
        
        # Parse fixed assignments
        locked_seats_per_round = defaultdict(dict)
        
        # Track names used in each round for duplicate detection
        names_per_round = defaultdict(list)
        
        # Track parsing errors
        parsing_errors = []
        
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
                    
                    # Check if this name is already used in this round
                    if name in names_per_round[round_idx]:
                        st.warning(f"Name '{name}' is assigned multiple times in round {round_num}.")
                    
                    # Add to tracking collections
                    locked_seats_per_round[round_idx][seat_id] = name
                    names_per_round[round_idx].append(name)
            except Exception as e:
                parsing_errors.append(f"Error parsing line: {line} - {str(e)}")
        
        # Show parsing errors at the end
        for error in parsing_errors:
            st.error(error)
        
        # Check if any assigned names are not in the guest list
        if hasattr(st.session_state, 'guests'):
            guests_set = set(st.session_state.guests)
            unknown_guests = []
            
            for round_idx, assignments in locked_seats_per_round.items():
                for seat_id, name in assignments.items():
                    if name not in guests_set:
                        unknown_guests.append((round_idx + 1, seat_id, name))
            
            if unknown_guests:
                st.error("The following locked assignments use names not in the guest list:")
                for round_num, seat_id, name in unknown_guests:
                    st.error(f"Round {round_num}, Seat {seat_id}: '{name}'")
        
        st.session_state[str(Settings.FIXED_ASSIGNMENTS)+_FROM_INPUT] = dict(locked_seats_per_round)

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

def set_weights():
    with st.sidebar.expander("Weights"):
        st.markdown("""
            Adjust the weights for different optimization criteria. Higher values give more importance to that criterion.
        """)
        
        settings = load_settings()
        weights_dict = settings.get(str(Settings.WEIGHTS), {}) if "weights" in settings else {
            str(Weights.PREFERRED_NEIGHBOR): 20.0,
            str(Weights.EXCLUDED_NEIGHBOR): 50.0,
            str(Weights.GENDER_ALTERNATION): 150.0,
            str(Weights.REPEAT_NEIGHBOR): 15.0,
            str(Weights.CORNER): 10.0
        }
        
        # Create sliders for each weight
        updated_weights = {}
        
        # Use the correct keys from the settings file
        preferred_value = int(weights_dict.get(str(Weights.PREFERRED_NEIGHBOR), 20))
        updated_weights[str(Weights.PREFERRED_NEIGHBOR)] = st.slider(
            "Preferred Neighbor", 
            min_value=0, 
            max_value=200, 
            value=preferred_value,
            step=1,
            help="Higher values give more importance to seating people next to their preferred neighbors."
        )
        
        excluded_value = int(weights_dict.get(Weights.EXCLUDED_NEIGHBOR, 50))
        updated_weights[str(Weights.EXCLUDED_NEIGHBOR)] = st.slider(
            "Excluded Neighbors", 
            min_value=0, 
            max_value=200, 
            value=excluded_value,
            step=1,
            help="Higher values give more importance to avoiding seating people next to their excluded neighbors."
        )
        
        gender_value = int(weights_dict.get(Weights.GENDER_ALTERNATION, 150))
        updated_weights[str(Weights.GENDER_ALTERNATION)] = st.slider(
            "Gender Alternation", 
            min_value=0, 
            max_value=200, 
            value=gender_value,
            step=1,
            help="Higher values give more importance to alternating genders at the table."
        )
        
        repeat_value = int(weights_dict.get(Weights.REPEAT_NEIGHBOR, 150))
        updated_weights[str(Weights.REPEAT_NEIGHBOR)] = st.slider(
            "Repeated Neighbor", 
            min_value=0, 
            max_value=200, 
            value=repeat_value,
            step=1,
            help="Higher values give more importance to avoiding repeat neighbors across different rounds."
        )
        
        corner_value = int(weights_dict.get(Weights.CORNER, 10))
        updated_weights[str(Weights.CORNER)] = st.slider(
            "Repeated corner position", 
            min_value=0, 
            max_value=200, 
            value=corner_value,
            step=1,
            help="Higher values give more importance to avoiding the same person sitting in corner positions repeatedly."
        )
        
        # Store the updated weights in session state
        st.session_state["weights"] = updated_weights

def visualize_guest_seating(arrangements, tables, table_letters):
    """
    Visualize seating arrangements for a specific guest across all rounds.
    Highlights the selected guest, their immediate neighbors, and future interactions.
    """
    st.subheader("Guest Seating Visualization")
    
    seat_neighbors = compute_seat_neighbors(tables)
    
    # Get all unique guests from arrangements
    all_guests = set()
    for arrangement in arrangements:
        all_guests.update(arrangement.values())
    
    # Sort guests alphabetically for the dropdown
    sorted_guests = sorted(list(all_guests))
    
    # Create a dropdown to select a guest
    selected_guest = st.selectbox("Select a guest to visualize:", sorted_guests)
    
    if selected_guest:
        # Create tabs for each round
        round_tabs = st.tabs([f"Round {i+1}" for i in range(len(arrangements))])
        
        # Track all neighbors across rounds for the selected guest
        all_neighbors = set()
        guest_positions = {}
        
        # First pass to collect all neighbors
        for round_idx, arrangement in enumerate(arrangements):
            # Find the seat of the selected guest in this round
            guest_seat = None
            for seat, guest in arrangement.items():
                if guest == selected_guest:
                    guest_seat = seat
                    guest_positions[round_idx] = seat
                    break
            
            if guest_seat:
                # Get immediate neighbors in this round
                for neighbor_seat in seat_neighbors[guest_seat]["all"]:
                    if neighbor_seat in arrangement:
                        neighbor = arrangement[neighbor_seat]
                        all_neighbors.add(neighbor)
        
        # Now create visualizations for each round
        for round_idx, arrangement in enumerate(arrangements):
            with round_tabs[round_idx]:
                # Create a grid for each table
                table_cols = st.columns(len(tables))
                
                for table_id in sorted(tables.keys()):
                    with table_cols[table_id]:
                        table_letter = table_letters[table_id]
                        st.markdown(f"**Table {table_letter}**")
                        
                        # Create a DataFrame to represent the table
                        table_data = []
                        for row in [0, 1]:
                            for col in range(tables[table_id]):
                                seat = (table_id, row, col)
                                guest = arrangement.get(seat, "")
                                
                                # Determine the status of this guest for coloring
                                status = "normal"
                                if guest == selected_guest:
                                    status = "selected"
                                elif guest_positions.get(round_idx) and seat in seat_neighbors[guest_positions[round_idx]]["all"]:
                                    status = "immediate_neighbor"
                                elif guest in all_neighbors and guest != selected_guest:
                                    status = "future_neighbor"
                                
                                seat_label = f"{table_letter}{col+1}" if row == 0 else f"{table_letter}{col+1+tables[table_id]}"
                                
                                table_data.append({
                                    "Seat": seat_label,
                                    "Guest": guest,
                                    "Status": status
                                })
                        
                        # Create a DataFrame and style it
                        df = pd.DataFrame(table_data)
                        
                        # Apply styling based on status
                        def highlight_cells(row):
                            if row["Status"] == "selected":
                                return ["background-color: #ffcc00"] * len(row)
                            elif row["Status"] == "immediate_neighbor":
                                return ["background-color: #99ff99"] * len(row)
                            elif row["Status"] == "future_neighbor":
                                return ["background-color: #ccccff"] * len(row)
                            return [""] * len(row)
                        
                        styled_df = df.style.apply(highlight_cells, axis=1)
                        st.dataframe(styled_df, hide_index=True)
                
                # Add a legend
                st.markdown("""
                **Legend:**
                - <span style='background-color: #ffcc00; padding: 2px 5px;'>Selected Guest</span>
                - <span style='background-color: #99ff99; padding: 2px 5px;'>Immediate Neighbor (this round)</span>
                - <span style='background-color: #ccccff; padding: 2px 5px;'>Neighbor in Other Rounds</span>
                """, unsafe_allow_html=True)
                
                # Show statistics for this round
                if round_idx in guest_positions:
                    guest_seat = guest_positions[round_idx]
                    immediate_neighbors = []
                    
                    for neighbor_seat in seat_neighbors[guest_seat]["all"]:
                        if neighbor_seat in arrangement:
                            neighbor = arrangement[neighbor_seat]
                            immediate_neighbors.append(neighbor)
                    
                    st.markdown(f"**Round {round_idx+1} Details:**")
                    table_letter = table_letters[guest_seat[0]]
                    row, col = guest_seat[1], guest_seat[2]
                    seat_num = col+1 if row == 0 else col+1+tables[guest_seat[0]]
                    st.markdown(f"- Seated at: Table {table_letter}, Seat {table_letter}{seat_num}")
                    st.markdown(f"- Immediate neighbors: {', '.join(immediate_neighbors)}")

def show_arrangement_overview(arrangements, tables, table_letters):            
            # Add the neighbor summary expander
            with st.expander("Overall Neighbour Summary"):
                data = []
                preferred_neighbors = st.session_state.get(str(Settings.PREFERRED_NEIGHBORS)+_FROM_INPUT, {})
                excluded_neighbors = st.session_state.get(str(Settings.EXCLUDED_NEIGHBORS)+_FROM_INPUT, {})
                
                # Track seats for each person across rounds
                person_seats = defaultdict(list)
                for round_idx, arrangement in enumerate(arrangements):
                    for seat, person in arrangement.items():
                        table_letter = table_letters[seat[0]]
                        row, col = seat[1], seat[2]
                        seat_num = col + 1 if row == 0 else col + 1 + tables[seat[0]]
                        # Store tuple of (round_idx, seat_label) for sorting
                        person_seats[person].append((round_idx, f"{table_letter}{seat_num}"))
                
                # Sort by round_idx and convert to comma-separated string of just the seat labels
                for person in person_seats:
                    sorted_seats = sorted(person_seats[person], key=lambda x: x[0])
                    person_seats[person] = ", ".join(seat_label for _, seat_label in sorted_seats)
                
                for person, types_dict in st.session_state.neighbors_info.items():
                    side_neighbors = set(types_dict["side"])
                    preferred_count = len(side_neighbors.intersection(set(preferred_neighbors.get(person, []))))
                    excluded_count = len(side_neighbors.intersection(set(excluded_neighbors.get(person, []))))
                    
                    data.append({
                        "Person": person,
                        "Seats": person_seats[person],
                        "Side Neighbours": ", ".join(sorted(types_dict["side"])),
                        "Front Neighbours": ", ".join(sorted(types_dict["front"])),
                        "Preferred": ", ".join(sorted(preferred_neighbors.get(person, []))),
                        "Excluded": ", ".join(sorted(excluded_neighbors.get(person, []))),
                        "Preferred Count": preferred_count,
                        "Excluded Count": excluded_count,
                        "Diagonal Neighbours": ", ".join(sorted(types_dict["diagonal"])),
                        "Corner Count": st.session_state.corner_count.get(person, 0),
                        "Gender": st.session_state.person_genders.get(person, "X"),

                    })
                nbr_df = pd.DataFrame(data)
                def color_preferred_count(val):
                    return 'background-color: #006400; color: white' if val > 0 else ''
                
                def color_excluded_count(val):
                    return 'background-color: #8B0000; color: white' if val > 0 else ''
                
                styled_df = nbr_df.style.applymap(color_preferred_count, subset=['Preferred Count'])
                styled_df = styled_df.applymap(color_excluded_count, subset=['Excluded Count'])
                st.dataframe(styled_df, height=400)

def compute_neighbors_info(arrangements, tables, table_letters):
    # Initialize neighbor tracking
    neighbors_info = defaultdict(lambda: defaultdict(set))
    corner_count = defaultdict(int)
    
    # Track corners for each table
    for table_id, seats_per_side in tables.items():
        for arrangement_idx, arrangement in enumerate(arrangements):
            # Corner positions for this table
            corners = [
                (table_id, 0, 0),  # Top-left
                (table_id, 0, seats_per_side - 1),  # Top-right
                (table_id, 1, 0),  # Bottom-left
                (table_id, 1, seats_per_side - 1)  # Bottom-right
            ]
            
            # Count corner occurrences
            for corner in corners:
                if corner in arrangement:
                    person = arrangement[corner]
                    corner_count[person] += 1
            
            # Track neighbors
            seat_neighbors = compute_seat_neighbors(tables)
            for seat, person in arrangement.items():
                for neighbor_seat in seat_neighbors[seat]["side"]:
                    if neighbor_seat in arrangement:
                        neighbors_info[person]["side"].add(arrangement[neighbor_seat])
                for neighbor_seat in seat_neighbors[seat]["front"]:
                    if neighbor_seat in arrangement:
                        neighbors_info[person]["front"].add(arrangement[neighbor_seat])
                for neighbor_seat in seat_neighbors[seat]["diagonal"]:
                    if neighbor_seat in arrangement:
                        neighbors_info[person]["diagonal"].add(arrangement[neighbor_seat])
    
    # Store in session state for the summary
    st.session_state.neighbors_info = neighbors_info
    st.session_state.corner_count = corner_count

def main():
    st.set_page_config(layout="wide")

    st.title("SeatPlan v2")
    st.markdown("6th March 2025 - 00:10")
    
    import_export_settings()
    
    set_tables()
    set_guests()
    set_preferred_neighbors()
    set_excluded_neighbors()
    set_fixed_assignments()
    set_same_gender_ok()
    set_groups()
    set_weights()

    
    TABLES, TABLE_LETTERS = parse_table_definitions(st.session_state[str(Settings.TABLE_DEFINITIONS)])

    guests = st.session_state.guests
    
    st.markdown("## Seating Generation")
    cols = st.columns([1, 1, 5])  
    with cols[0]:
        num_rounds = st.number_input("Number of arrangements", min_value=1, max_value=5, value=3, step=1)
    with cols[1]:
        use_optimization = st.checkbox("Use optimization", value=True)
    
    with cols[2]:
        max_iterations = st.number_input("Optimization iterations", min_value=0, max_value=100000, value=5000, step=100)


    # Check if we have previous arrangements to build upon
    previous_arrangements = None
    if "previous_arrangements" in st.session_state:
        use_previous = st.checkbox("Build upon previous arrangements", value=False)
        if use_previous:
            previous_arrangements = st.session_state.previous_arrangements
    
    run_button = st.button("Generate Seating Arrangements", key="generate_button", type="primary")
    
    if run_button or "arrangements" not in st.session_state:
        
        # Get preferences
        preferred_neighbors = st.session_state[str(Settings.PREFERRED_NEIGHBORS)+_FROM_INPUT] if hasattr(st.session_state, str(Settings.PREFERRED_NEIGHBORS)+_FROM_INPUT) else {}
        excluded_neighbors = st.session_state[str(Settings.EXCLUDED_NEIGHBORS)+_FROM_INPUT] if hasattr(st.session_state, str(Settings.EXCLUDED_NEIGHBORS)+_FROM_INPUT) else {}
        same_gender_ok = st.session_state[str(Settings.SAME_GENDER_OK)+_FROM_INPUT] if hasattr(st.session_state, str(Settings.SAME_GENDER_OK)+_FROM_INPUT) else []
        repeat_ok_groups = st.session_state[str(Settings.REPEAT_OK_GROUPS)+_FROM_INPUT] if hasattr(st.session_state, str(Settings.REPEAT_OK_GROUPS)+_FROM_INPUT) else []
        
        # Generate arrangements
        locked_seats_per_round = st.session_state[str(Settings.FIXED_ASSIGNMENTS)+_FROM_INPUT] if hasattr(st.session_state, str(Settings.FIXED_ASSIGNMENTS)+_FROM_INPUT) else {}
        
        arrangements = run_optimization(
            TABLES, 
            guests, 
            num_rounds, 
            locked_seats_per_round, 
            table_letters=TABLE_LETTERS,
            previous_arrangements=previous_arrangements,
            person_genders=st.session_state.person_genders if use_optimization else None,
            preferred_neighbors=preferred_neighbors if use_optimization else None,
            excluded_neighbors=excluded_neighbors if use_optimization else None,
            same_gender_ok=same_gender_ok if use_optimization else None,
            repeat_ok_groups=repeat_ok_groups if use_optimization else None,
            max_iterations=max_iterations
        )
        
        # Store arrangements for potential future use
        st.session_state.previous_arrangements = arrangements
        st.session_state.arrangements = arrangements
    
    # Display the current arrangements
    if "arrangements" in st.session_state:
        with st.spinner("Rendering seating arrangements..."):
            st.markdown("## Arrangements")
            compute_neighbors_info(st.session_state.arrangements, TABLES, TABLE_LETTERS)
            show_arrangement_overview(st.session_state.arrangements, TABLES, TABLE_LETTERS)
            show_arrangements(st.session_state.arrangements, TABLES, TABLE_LETTERS)

    # Visualize seating for a specific guest
    #visualize_guest_seating(st.session_state.arrangements, TABLES, TABLE_LETTERS)


if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
