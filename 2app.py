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

def set_guests():
    with st.sidebar.expander("Guests"):
        st.markdown("""
            Enter the names of the guests.
        """)
        if "uploaded_settings" in st.session_state:
            settings = st.session_state.uploaded_settings
            male_text = st.text_area("Males (one per line)", 
                                            value=get_setting(settings, Settings.MALES), height=150,
                                            key='male_text', help="One male name per line.")
            male_count = len([name for name in male_text.splitlines() if name.strip()])
            st.caption(f"Male count: {male_count}")
            
            female_text = st.text_area("Females (one per line)", 
                                            value=get_setting(settings, Settings.FEMALES), height=150,
                                            key='female_text', help="One female name per line.")
            female_count = len([name for name in female_text.splitlines() if name.strip()])
            st.caption(f"Female count: {female_count}")
        else:
            male_text = st.text_area("Males (one per line)", value=get_setting(DEFAULT_SETTINGS, Settings.MALES), height=150,
                                            key='male_text', help="One male name per line.")
            male_count = len([name for name in male_text.splitlines() if name.strip()])
            st.caption(f"Male count: {male_count}")
            
            female_text = st.text_area("Females (one per line)", value=get_setting(DEFAULT_SETTINGS, Settings.FEMALES), height=150,
                                            key='female_text', help="One female name per line.")
            female_count = len([name for name in female_text.splitlines() if name.strip()])
            st.caption(f"Female count: {female_count}")
            
        male_names = parse_lines(male_text)
        female_names = parse_lines(female_text)
        guests = male_names + female_names
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
            if "uploaded_settings" in st.session_state:
                settings = st.session_state.uploaded_settings
                st.text_area(
                    "Define tables (e.g., 'A: 8')", 
                    value=get_setting(settings, Settings.TABLE_DEFINITIONS),
                    height=150,
                    key=str(Settings.TABLE_DEFINITIONS),
                    help="Each line: Letter: Number"
                )
            else:
                st.text_area(
                    "Define tables (e.g., 'A: 8')", 
                    value=get_setting(DEFAULT_SETTINGS, Settings.TABLE_DEFINITIONS),
                    height=150,
                    key=str(Settings.TABLE_DEFINITIONS),
                    help="Each line: Letter: Number"
                ) 
                
        TABLES, TABLE_LETTERS = parse_table_definitions(st.session_state.table_definitions)
        
        with col2:
            for table_id in sorted(TABLES.keys()):
                table_letter = TABLE_LETTERS[table_id]
                html = generate_table_html(table_id, table_letter, TABLES)
                # Calculate dynamic height based on number of seats
                table_height = max(150, 100 + (TABLES[table_id] // 8) * 50)
                components.html(html, height=table_height, scrolling=True)


def run_optimization(tables, guests, num_rounds=3):
    """
    Args:
        tables: {table_id: seats_per_side}
        guests: list of guest names
        num_rounds: number of rounds
    
    Returns:
        List of arrangements, where each arrangement is a dictionary mapping seats to guest names
    """
    # Generate all seats
    seats = generate_seats(tables)
    
    num_seats = len(seats)    
    # Generate random arrangements
    arrangements = []
    for _ in range(num_rounds):
        # Shuffle the guests
        random_guests = copy.deepcopy(guests)
        random.shuffle(random_guests)
        
        # Create arrangement mapping seats to guests
        arrangement = {}
        for i, seat in enumerate(seats):
            if i < len(random_guests):
                arrangement[seat] = random_guests[i]
        
        arrangements.append(arrangement)
    
    return arrangements


def main():
    st.set_page_config(layout="wide")

    st.title("SeatPlan v2")
    
    set_tables()
    

    set_guests()
    
    TABLES, TABLE_LETTERS = parse_table_definitions(st.session_state.table_definitions)

    
    ### Show arrangements
    
    st.markdown("## Arrangements") 
    
    # Place inputs side by side with smaller widths
    cols = st.columns([1, 1, 5])  # Adjust the ratio to make inputs smaller
    with cols[0]:
        num_rounds = st.number_input("Number of arrangements", min_value=1, max_value=10, value=3)
    with cols[1]:
        tables_per_row = st.number_input("Tables per row", min_value=1, max_value=5, value=2, step=1)

    guests = st.session_state.guests
    
    # Generate arrangements
    arrangements = run_optimization(TABLES, guests, num_rounds)
    
    # Display each arrangement
    for round_idx, arrangement in enumerate(arrangements):
        st.markdown(f"### Arrangement {round_idx + 1}")
        
        # Create columns for each table
        cols = st.columns(len(TABLES))
        
        # Display each table
        table_ids = sorted(TABLES.keys())

        for i in range(0, len(table_ids), tables_per_row):
            cols = st.columns(tables_per_row)
            for j, table_id in enumerate(table_ids[i:i+tables_per_row]):
                with cols[j]:
                    table_letter = TABLE_LETTERS[table_id]
                    st.markdown(f"**Table {table_letter}**")
                    html = generate_arrangement_html(arrangement, table_id, TABLES)
                    # Calculate dynamic height and width based on number of seats
                    table_height = max(150, 100 + (TABLES[table_id] // 8) * 50)
                    calculated_width = TABLES[table_id] * 62  # Adjust this multiplier if needed
                    components.html(html, height=table_height, width=calculated_width, scrolling=True)

                # table_letter = TABLE_LETTERS[table_id]
                # st.markdown(f"**Table {table_letter}**")
                # html = generate_arrangement_html(arrangement, table_id, TABLES)
                # # Calculate dynamic height based on number of seats
                # table_height = max(150, 100 + (TABLES[table_id] // 8) * 50)
                # components.html(html, height=table_height, scrolling=True)


if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
