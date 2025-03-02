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

def run_optimization(tables, guests, num_rounds=3, locked_seats_per_round=None, table_letters=None):
    """
    Args:
        tables: {table_id: seats_per_side}
        guests: list of guest names
        num_rounds: number of rounds
        locked_seats_per_round:
    Returns:
        [{(table_id, row, col): guest_name}]
    """
    # Generate all seats
    seats = generate_seats(tables)
    
    # Generate random arrangements
    arrangements = []
    
    for round_index in range(num_rounds):
        arrangement = {}
        # First, if there are locked seats for this round, place them.
        if locked_seats_per_round and round_index in locked_seats_per_round:
            locked_assignments = locked_seats_per_round[round_index]  # e.g., {"A1": "Johan", "A2": "Ludmilla"}
            for seat_label, guest in locked_assignments.items():
                seat_tuple = seat_label_to_tuple(seat_label, tables, table_letters)
                if seat_tuple:
                    arrangement[seat_tuple] = guest
    
        # remove locked guests
        locked_guests = set(arrangement.values())
        available_guests = [g for g in guests if g not in locked_guests]
        
        # place remaining guests
        random.shuffle(available_guests)

        # Fill in the remaining seats with the available guests.
        for seat in seats:
            if seat not in arrangement:
                if available_guests:
                    arrangement[seat] = available_guests.pop(0)
                else:
                    break
        
        arrangements.append(arrangement)
        
    return arrangements

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
    
    cols = st.columns([1, 1, 5])  
    with cols[0]:
        num_rounds = st.number_input("Number of arrangements", min_value=1, max_value=5, value=3, step=1)

    # Generate arrangements
    locked_seats_per_round = st.session_state.locked_seats_per_round if hasattr(st.session_state, 'locked_seats_per_round') else {}
    arrangements = run_optimization(TABLES, guests, num_rounds, locked_seats_per_round, table_letters=TABLE_LETTERS)
    
    show_arrangements(arrangements, TABLES, TABLE_LETTERS)
    
    st.write(str(arrangements))

    


if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
