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

def get_setting(settings_dict, setting_key):
    """Helper function to get a setting value using an enum key"""
    return settings_dict[str(setting_key)]

def tables():
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


def main():
    st.title("SeatPlan v2")
    
    tables()
    
    st.write(st.session_state.table_definitions)
    

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
