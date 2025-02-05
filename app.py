import streamlit as st
import streamlit.web.bootstrap
import streamlit.components.v1 as components
import pandas as pd
import random
import math
import copy
from collections import defaultdict
import json

#####################################
# 1. Table Definitions & Utilities  #
#####################################

DEFAULT_TABLE_DEF = "A: 3\nB: 3\nC: 4"

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

#####################################
# 2. Seat and Neighbor Utilities    #
#####################################

def generate_seats(tables):
    seats = []
    for t, seats_per_side in tables.items():
        for row in [0, 1]:
            for col in range(seats_per_side):
                seats.append((t, row, col))
    return seats

def compute_seat_neighbors(tables):
    seat_neighbors = {}
    for t, seats_per_side in tables.items():
        for row in [0, 1]:
            for col in range(seats_per_side):
                s = (t, row, col)
                neighbors = {
                    "side": [],
                    "front": [],
                    "diagonal": []
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
    return seat_neighbors

def parse_preferred_side_neighbours(text):
    pref = {}
    lines = text.strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        try:
            person_part, neighbours_str = line.split(":", 1)
            person = person_part.strip()
            neighbours = [n.strip() for n in neighbours_str.split(",") if n.strip()]
            pref[person] = neighbours
        except Exception:
            continue
    return pref

#####################################
# 3. Cost & Neighbor Preference      #
#####################################

def compute_cost(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                 side_weight=1.0, front_weight=1.0, diagonal_weight=1.0,
                 corner_weight=3.0, gender_weight=5.0, fixed_weight=2.0, empty_weight=5.0,
                 preferred_side_preferences=None, preferred_side_weight=1.0,
                 breakdown=False):
    """
    Computes the total cost over all rounds.
    If breakdown=True, returns (total_cost, breakdown_dict) where breakdown_dict contains:
      - side_cost, front_cost, diagonal_cost, neighbor_cost,
      - corner_cost, gender_cost, empty_cost, preferred_side_cost.
    Otherwise, returns total_cost.
    """
    # Gather neighbor lists and corner counts per person.
    person_neighbors_by_type = defaultdict(lambda: {"side": [], "front": [], "diagonal": []})
    person_corner_counts = defaultdict(int)
    for round_assign in assignments:
        for seat, person in round_assign.items():
            t, row, col = seat
            if col == 0 or col == tables[t] - 1:
                person_corner_counts[person] += 1
            for n_type, n_list in seat_neighbors[seat].items():
                for n_seat in n_list:
                    neighbor_person = round_assign[n_seat]
                    person_neighbors_by_type[person][n_type].append(neighbor_person)
                    
    # Neighbor cost (by type)
    side_cost = 0
    front_cost = 0
    diagonal_cost = 0
    for person, type_dict in person_neighbors_by_type.items():
        if person_genders.get(person, "X") == "X":
            continue
        multiplier = fixed_weight if person in fixed_positions else 1.0
        repeats_side = len(type_dict["side"]) - len(set(type_dict["side"]))
        repeats_front = len(type_dict["front"]) - len(set(type_dict["front"]))
        repeats_diag = len(type_dict["diagonal"]) - len(set(type_dict["diagonal"]))
        side_cost += multiplier * side_weight * repeats_side
        front_cost += multiplier * front_weight * repeats_front
        diagonal_cost += multiplier * diagonal_weight * repeats_diag
    neighbor_cost = side_cost + front_cost + diagonal_cost

    # Corner cost (per person)
    corner_count_total = 0
    for person, count in person_corner_counts.items():
        if person_genders.get(person, "X") == "X":
            continue
        if count > 1:
            corner_count_total += (count - 1)
    total_corner_cost = corner_weight * corner_count_total

    # Gender cost (for adjacent same-gender pairs)
    gender_count = 0
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
                        gender_count += 1
    total_gender_cost = gender_weight * gender_count

    # Empty seat clustering cost
    empty_count = 0
    for round_assign in assignments:
        for t, seats_per_side in tables.items():
            for row in [0, 1]:
                seats_in_row = [(t, row, col) for col in range(seats_per_side)]
                for i in range(len(seats_in_row) - 1):
                    p1 = round_assign[seats_in_row[i]]
                    p2 = round_assign[seats_in_row[i+1]]
                    if (p1.startswith("Empty") and not p2.startswith("Empty")) or (not p1.startswith("Empty") and p2.startswith("Empty")):
                        empty_count += 1
    total_empty_cost = empty_weight * empty_count

    # Preferred side neighbour cost
    pref_side_count = 0
    if preferred_side_preferences is None:
        preferred_side_preferences = {}
    for round_assign in assignments:
        for seat, person in round_assign.items():
            if person in preferred_side_preferences:
                desired = set(preferred_side_preferences[person])
                side_nbrs = [round_assign[n_seat] for n_seat in seat_neighbors[seat]["side"]]
                for d in desired:
                    if d not in side_nbrs:
                        pref_side_count += 1
    total_pref_side_cost = preferred_side_weight * pref_side_count

    total_cost = neighbor_cost + total_corner_cost + total_gender_cost + total_empty_cost + total_pref_side_cost

    if breakdown:
        breakdown_dict = {
            "side_cost": side_cost,
            "front_cost": front_cost,
            "diagonal_cost": diagonal_cost,
            "neighbor_cost": neighbor_cost,
            "corner_cost": total_corner_cost,
            "gender_cost": total_gender_cost,
            "empty_cost": total_empty_cost,
            "preferred_side_cost": total_pref_side_cost,
            "total_cost": total_cost
        }
        return total_cost, breakdown_dict
    else:
        return total_cost

def get_neighbors_info_by_type(assignments, seat_neighbors, tables):
    person_neighbors = defaultdict(lambda: {"side": set(), "front": set(), "diagonal": set()})
    corner_count = defaultdict(int)
    for round_assign in assignments:
        for seat, person in round_assign.items():
            t, row, col = seat
            if col == 0 or col == tables[t] - 1:
                corner_count[person] += 1
            for n_type, neighbor_list in seat_neighbors[seat].items():
                for n_seat in neighbor_list:
                    neighbor_person = round_assign[n_seat]
                    person_neighbors[person][n_type].add(neighbor_person)
    return person_neighbors, corner_count

#####################################
# 4. Optimization Functions          #
#####################################

def initialize_assignments(people, tables, fixed_positions, num_rounds=3):
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
                         side_weight=1.0, front_weight=1.0, diagonal_weight=1.0,
                         corner_weight=3.0, gender_weight=5.0, fixed_weight=2.0, empty_weight=5.0,
                         preferred_side_preferences=None, preferred_side_weight=1.0,
                         record_history=True):
    """
    Uses simulated annealing to optimize seating assignments.
    Optionally records a history of cost breakdowns every 100 iterations.
    Returns best_assignments, best_cost, and cost_history (a list of dicts keyed by iteration).
    """
    if record_history:
        current_cost, current_breakdown = compute_cost(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                                                       side_weight, front_weight, diagonal_weight, corner_weight,
                                                       gender_weight, fixed_weight, empty_weight,
                                                       preferred_side_preferences, preferred_side_weight,
                                                       breakdown=True)
    else:
        current_cost = compute_cost(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                                    side_weight, front_weight, diagonal_weight, corner_weight,
                                    gender_weight, fixed_weight, empty_weight,
                                    preferred_side_preferences, preferred_side_weight,
                                    breakdown=False)
    best_cost = current_cost
    best_assignments = copy.deepcopy(assignments)
    num_rounds = len(assignments)
    seats = generate_seats(tables)
    free_seats_by_round = []
    for _ in range(num_rounds):
        fixed = set(fixed_positions.values())
        free_seats_by_round.append([s for s in seats if s not in fixed])
    temp = initial_temp

    cost_history = []
    if record_history:
        cost_history.append({"iteration": 0, **current_breakdown})
    
    for iter_num in range(1, iterations + 1):
        r = random.randint(0, num_rounds - 1)
        free_seats = free_seats_by_round[r]
        seat1, seat2 = random.sample(free_seats, 2)
        assignments[r][seat1], assignments[r][seat2] = assignments[r][seat2], assignments[r][seat1]
        if record_history:
            new_cost, new_breakdown = compute_cost(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                                                   side_weight, front_weight, diagonal_weight, corner_weight,
                                                   gender_weight, fixed_weight, empty_weight,
                                                   preferred_side_preferences, preferred_side_weight,
                                                   breakdown=True)
        else:
            new_cost = compute_cost(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                                    side_weight, front_weight, diagonal_weight, corner_weight,
                                    gender_weight, fixed_weight, empty_weight,
                                    preferred_side_preferences, preferred_side_weight,
                                    breakdown=False)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_cost = new_cost
            if record_history:
                current_breakdown = new_breakdown
            if new_cost < best_cost:
                best_cost = new_cost
                best_assignments = copy.deepcopy(assignments)
        else:
            assignments[r][seat1], assignments[r][seat2] = assignments[r][seat2], assignments[r][seat1]
        temp *= cooling_rate

        if record_history and (iter_num % 100 == 0):
            # Record the cost breakdown every 100 iterations.
            cost_history.append({"iteration": iter_num, **current_breakdown})
    if record_history:
        return best_assignments, best_cost, cost_history
    else:
        return best_assignments, best_cost

#####################################
# 4b. Per-Person Cost Breakdown       #
#####################################

def compute_individual_cost_breakdown(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                                      preferred_side_preferences, weights):
    """
    Computes a cost breakdown per person from the best assignments.
    The returned dict maps person -> dict with keys:
       side_cost, front_cost, diagonal_cost, neighbor_cost,
       corner_cost, preferred_side_cost, gender_cost, empty_cost, total_cost.
       
    Note: Gender and empty costs are attributed by splitting the pair cost.
    """
    # Unpack weight parameters
    side_weight = weights.get("side_neighbour_weight", 1.0)
    front_weight = weights.get("front_neighbour_weight", 1.0)
    diagonal_weight = weights.get("diagonal_neighbour_weight", 1.0)
    corner_weight = weights.get("corner_weight", 3.0)
    gender_weight = weights.get("gender_weight", 5.0)
    fixed_weight = weights.get("fixed_weight", 2.0)
    empty_weight = weights.get("empty_weight", 5.0)
    preferred_side_weight = weights.get("preferred_side_weight", 1.0)
    
    # Initialize dictionaries.
    indiv_breakdown = {}
    for person in person_genders:
        indiv_breakdown[person] = {
            "side_cost": 0,
            "front_cost": 0,
            "diagonal_cost": 0,
            "neighbor_cost": 0,
            "corner_cost": 0,
            "preferred_side_cost": 0,
            "gender_cost": 0,
            "empty_cost": 0,
            "total_cost": 0
        }
    # Compute neighbor cost per person
    indiv_neighbors = defaultdict(lambda: {"side": [], "front": [], "diagonal": []})
    for round_assign in assignments:
        for seat, person in round_assign.items():
            for n_type, n_list in seat_neighbors[seat].items():
                for n_seat in n_list:
                    indiv_neighbors[person][n_type].append(round_assign[n_seat])
    for person, nbrs in indiv_neighbors.items():
        if person_genders.get(person, "X") == "X":
            continue
        multiplier = fixed_weight if person in fixed_positions else 1.0
        repeats_side = len(nbrs["side"]) - len(set(nbrs["side"]))
        repeats_front = len(nbrs["front"]) - len(set(nbrs["front"]))
        repeats_diag = len(nbrs["diagonal"]) - len(set(nbrs["diagonal"]))
        cost_side = multiplier * side_weight * repeats_side
        cost_front = multiplier * front_weight * repeats_front
        cost_diag = multiplier * diagonal_weight * repeats_diag
        indiv_breakdown[person]["side_cost"] = cost_side
        indiv_breakdown[person]["front_cost"] = cost_front
        indiv_breakdown[person]["diagonal_cost"] = cost_diag
        indiv_breakdown[person]["neighbor_cost"] = cost_side + cost_front + cost_diag

    # Corner cost per person
    corner_count = defaultdict(int)
    for round_assign in assignments:
        for seat, person in round_assign.items():
            t, row, col = seat
            if col == 0 or col == tables[t] - 1:
                corner_count[person] += 1
    for person, count in corner_count.items():
        if person_genders.get(person, "X") == "X":
            continue
        if count > 1:
            indiv_breakdown[person]["corner_cost"] = (count - 1) * corner_weight

    # Preferred side neighbour cost per person
    if preferred_side_preferences is None:
        preferred_side_preferences = {}
    for round_assign in assignments:
        for seat, person in round_assign.items():
            if person in preferred_side_preferences:
                desired = set(preferred_side_preferences[person])
                side_nbrs = [round_assign[n_seat] for n_seat in seat_neighbors[seat]["side"]]
                missing = sum(1 for d in desired if d not in side_nbrs)
                indiv_breakdown[person]["preferred_side_cost"] += missing * preferred_side_weight

    # Gender cost: For each adjacent same-gender pair, attribute half cost to each person.
    for round_assign in assignments:
        for t, seats_per_side in tables.items():
            for row in [0, 1]:
                seats_in_row = [(t, row, col) for col in range(seats_per_side)]
                for i in range(len(seats_in_row) - 1):
                    s1, s2 = seats_in_row[i], seats_in_row[i+1]
                    p1, p2 = round_assign[s1], round_assign[s2]
                    g1 = person_genders.get(p1, "X")
                    g2 = person_genders.get(p2, "X")
                    if g1 == "X" or g2 == "X":
                        continue
                    if g1 == g2:
                        indiv_breakdown[p1]["gender_cost"] += gender_weight / 2
                        indiv_breakdown[p2]["gender_cost"] += gender_weight / 2
    # Empty cost: For each adjacent pair where one is empty and one is not, assign the cost to the non-empty person.
    for round_assign in assignments:
        for t, seats_per_side in tables.items():
            for row in [0, 1]:
                seats_in_row = [(t, row, col) for col in range(seats_per_side)]
                for i in range(len(seats_in_row) - 1):
                    p1 = round_assign[seats_in_row[i]]
                    p2 = round_assign[seats_in_row[i+1]]
                    if p1.startswith("Empty") and not p2.startswith("Empty"):
                        indiv_breakdown[p2]["empty_cost"] += empty_weight
                    elif p2.startswith("Empty") and not p1.startswith("Empty"):
                        indiv_breakdown[p1]["empty_cost"] += empty_weight

    # Total cost per person (sum of the attributable components)
    for person, comp in indiv_breakdown.items():
        comp["total_cost"] = comp["neighbor_cost"] + comp["corner_cost"] + comp["preferred_side_cost"] + comp["gender_cost"] + comp["empty_cost"]

    return indiv_breakdown

#####################################
# 5. Seating Arrangement Display     #
#####################################

def seating_dataframe_for_table(assignments, table_id, table_letter):
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
# 6. Fixed Seat Input Parser         #
#####################################

def parse_fixed_seats(text):
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
            seat_num = int(seat_id[1:]) - 1  # 0-based index
            
            # Find table_id from table letter
            table_id = None
            for tid, letter in TABLE_LETTERS.items():
                if letter == table_letter:
                    table_id = tid
                    break
            if table_id is None:
                continue
                
            seats_per_side = TABLES[table_id]
            row = 1 if seat_num >= seats_per_side else 0
            col = seat_num % seats_per_side
            
            fixed[name] = (table_id, row, col)
        except Exception:
            continue
    return fixed

#####################################
# 7. Table Layout Visualization       #
#####################################

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
# 8. Run Optimization & Build Data  #
#####################################

def run_optimization_and_build_data(iterations, initial_temp, cooling_rate,
                                      side_weight, front_weight, diagonal_weight,
                                      corner_weight, gender_weight, fixed_weight, empty_weight,
                                      people, person_genders, fixed_positions, tables,
                                      preferred_side_preferences, preferred_side_weight):
    """
    Runs the seating optimization and returns:
      - best_assignments: list (per round) of seat -> person assignments,
      - best_cost: final cost,
      - neighbors_info: dict mapping each person to a dict of neighbour types,
      - corner_count: dict mapping each person to the number of rounds they sat at a corner,
      - cost_history: list of cost breakdowns over iterations.
    
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
    best_assignments, best_cost, cost_history = optimize_assignments(
        assignments, seat_neighbors, tables, fixed_positions, person_genders,
        iterations, initial_temp, cooling_rate,
        side_weight, front_weight, diagonal_weight, corner_weight, gender_weight, fixed_weight, empty_weight,
        preferred_side_preferences, preferred_side_weight,
        record_history=True
    )
    neighbors_info, corner_count = get_neighbors_info_by_type(best_assignments, seat_neighbors, tables)
    return best_assignments, best_cost, neighbors_info, corner_count, cost_history

def combine_all_seating_dataframes(assignments, tables, table_letters):
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
# 9. Main App                        #
#####################################

def get_current_settings():
    # Gather current settings from session state
    table_def_text = st.session_state.table_def_text if 'table_def_text' in st.session_state else DEFAULT_TABLE_DEF
    default_male = "John\nMike\nDavid\nSteve\nRobert\nJames\nWilliam\nRichard\nJoseph\nThomas"
    default_female = "Mary\nLinda\nSusan\nKaren\nPatricia\nBarbara\nNancy\nLisa\nBetty\nMargaret"
    male_text = st.session_state.male_text if 'male_text' in st.session_state else default_male
    female_text = st.session_state.female_text if 'female_text' in st.session_state else default_female
    fixed_text = st.session_state.fixed_text if 'fixed_text' in st.session_state else "John: A1\nMary: B2"
    pref_side_text = st.session_state.pref_side_text if 'pref_side_text' in st.session_state else ""
    iterations = st.session_state.iterations if 'iterations' in st.session_state else 20000
    initial_temp = st.session_state.initial_temp if 'initial_temp' in st.session_state else 10.0
    cooling_rate = st.session_state.cooling_rate if 'cooling_rate' in st.session_state else 0.9995
    side_weight = st.session_state.side_weight if 'side_weight' in st.session_state else 1.0
    front_weight = st.session_state.front_weight if 'front_weight' in st.session_state else 1.0
    diagonal_weight = st.session_state.diagonal_weight if 'diagonal_weight' in st.session_state else 1.0
    corner_weight = st.session_state.corner_weight if 'corner_weight' in st.session_state else 3.0
    gender_weight = st.session_state.gender_weight if 'gender_weight' in st.session_state else 5.0
    fixed_weight = st.session_state.fixed_weight if 'fixed_weight' in st.session_state else 2.0
    empty_weight = st.session_state.empty_weight if 'empty_weight' in st.session_state else 5.0
    preferred_side_weight = st.session_state.preferred_side_weight if 'preferred_side_weight' in st.session_state else 1.0
    
    return {
        "table_definitions": table_def_text,
        "male_names": male_text,
        "female_names": female_text,
        "fixed_assignments": fixed_text,
        "preferred_side_preferences_text": pref_side_text,
        "optimization_params": {
            "iterations": iterations,
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
        },
        "weights": {
            "side_neighbour_weight": side_weight,
            "front_neighbour_weight": front_weight,
            "diagonal_neighbour_weight": diagonal_weight,
            "corner_weight": corner_weight,
            "gender_weight": gender_weight,
            "fixed_weight": fixed_weight,
            "empty_weight": empty_weight,
            "preferred_side_weight": preferred_side_weight
        }
    }

def main():
    st.title("SeatPlan")
    st.markdown("##### Optimizing seating arrangements for events.")
    st.markdown("###### Use the sidebar for additional customization")
    
    # Sidebar: Settings download/upload
    st.sidebar.markdown("# Settings")
    if st.sidebar.button("Download Settings"):
        settings = get_current_settings()
        settings_json = json.dumps(settings, indent=2)
        st.sidebar.write("Settings ready for download:")
        st.sidebar.json(settings)
        st.sidebar.download_button(
            label="👉 Click here to download Settings JSON",
            data=settings_json,
            file_name="seatplan_settings.json",
            mime="application/json",
            key="settings_download"
        )
    uploaded_file = st.sidebar.file_uploader("Upload Settings", type=['json'])
    if uploaded_file is not None:
        try:
            settings = json.load(uploaded_file)
            st.session_state.uploaded_settings = settings
            st.sidebar.success("Settings loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading settings: {str(e)}")
    
    # Table Layout Configuration
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
                help="Each line: Letter: Number"
            )
        else:
            table_def_text = st.text_area(
                "Define tables (e.g., 'A: 8')", 
                value=DEFAULT_TABLE_DEF,
                height=150,
                key='table_def_text',
                help="Each line: Letter: Number"
            )
    global TABLES, TABLE_LETTERS
    TABLES, TABLE_LETTERS = parse_table_definitions(table_def_text)
    with col2:
        st.markdown("Seat numberings for each table. Corner seats are highlighted in light red.")
        for table_id in sorted(TABLES.keys()):
            table_letter = TABLE_LETTERS[table_id]
            html = generate_table_html(table_id, table_letter, TABLES)
            components.html(html, height=180, scrolling=False)
    
    # Guests
    st.sidebar.header("Guests")
    if "uploaded_settings" in st.session_state:
        settings = st.session_state.uploaded_settings
        male_text = st.sidebar.text_area("Males (one per line)", 
                                         value=settings["male_names"], height=150,
                                         key='male_text', help="One male name per line.")
        female_text = st.sidebar.text_area("Females (one per line)", 
                                           value=settings["female_names"], height=150,
                                           key='female_text', help="One female name per line.")
    else:
        default_male = "John\nMike\nDavid\nSteve\nRobert\nJames\nWilliam\nRichard\nJoseph\nThomas"
        default_female = "Mary\nLinda\nSusan\nKaren\nPatricia\nBarbara\nNancy\nLisa\nBetty\nMargaret"
        male_text = st.sidebar.text_area("Males (one per line)", value=default_male, height=150,
                                         key='male_text', help="One male name per line.")
        female_text = st.sidebar.text_area("Females (one per line)", value=default_female, height=150,
                                           key='female_text', help="One female name per line.")
    male_names = [name.strip() for name in male_text.splitlines() if name.strip()]
    female_names = [name.strip() for name in female_text.splitlines() if name.strip()]
    people = male_names + female_names
    person_genders = {}
    for name in male_names:
        person_genders[name] = "M"
    for name in female_names:
        person_genders[name] = "F"
    
    # Fixed seat assignments
    st.sidebar.header("Fixed Seat Assignments")
    if "uploaded_settings" in st.session_state:
        settings = st.session_state.uploaded_settings
        fixed_text = st.sidebar.text_area("Enter fixed seat assignments (e.g., 'John: A12')", 
                                          value=settings["fixed_assignments"], height=100,
                                          key='fixed_text', help="Format: Name: Seat")
    else:
        fixed_text = st.sidebar.text_area("Enter fixed seat assignments (e.g., 'John: A12')", 
                                          value="John: A1\nMary: B2", height=100,
                                          key='fixed_text', help="Format: Name: Seat")
    fixed_positions = parse_fixed_seats(fixed_text)
    
    # Preferred side neighbour preferences
    st.sidebar.header("Preferred Side Neighbour Preferences")
    pref_side_text = st.sidebar.text_area(
        "Format: Person: Neighbour1, Neighbour2, ...", 
        value=st.session_state.pref_side_text if 'pref_side_text' in st.session_state else "",
        height=100,
        key='pref_side_text',
        help="For example: Alice: Bob, Charlie"
    )
    preferred_side_preferences = parse_preferred_side_neighbours(pref_side_text)
    
    # Conditions & Weights
    st.sidebar.header("Conditions")
    st.sidebar.markdown("""
        Set the importance of each condition. Higher values make the condition more important.
    """)
    if "uploaded_settings" in st.session_state:
        settings = st.session_state.uploaded_settings
        side_weight = st.sidebar.number_input("Side Neighbour Weight", 
                                              value=settings["weights"].get("side_neighbour_weight", 5.0),
                                              step=1.0, format="%.1f",
                                              key='side_weight',
                                              help="Weight for repeated side neighbours.")
        front_weight = st.sidebar.number_input("Front Neighbour Weight", 
                                               value=settings["weights"].get("front_neighbour_weight", 2.0),
                                               step=1.0, format="%.1f",
                                               key='front_weight',
                                               help="Weight for repeated front neighbours.")
        diagonal_weight = st.sidebar.number_input("Diagonal Neighbour Weight", 
                                                  value=settings["weights"].get("diagonal_neighbour_weight", 1.0),
                                                  step=1.0, format="%.1f",
                                                  key='diagonal_weight',
                                                  help="Weight for repeated diagonal neighbours.")
        corner_weight = st.sidebar.number_input("Corner Weight", 
                                                value=settings["weights"]["corner_weight"], 
                                                step=0.1, format="%.1f",
                                                key='corner_weight',
                                                help="Weight for repeated corner seatings.")
        gender_weight = st.sidebar.number_input("Gender Weight", 
                                                value=settings["weights"]["gender_weight"], 
                                                step=0.1, format="%.1f",
                                                key='gender_weight',
                                                help="Weight for adjacent same-gender seats.")
        fixed_weight = st.sidebar.number_input("Fixed Seat Diversity Weight", 
                                               value=settings["weights"]["fixed_weight"], 
                                               step=0.1, format="%.1f",
                                               key='fixed_weight',
                                               help="Extra weight for fixed-seat persons.")
        empty_weight = st.sidebar.number_input("Empty Seat Clustering Weight", 
                                               value=settings["weights"]["empty_weight"], 
                                               step=0.1, format="%.1f",
                                               key='empty_weight',
                                               help="Weight for boundaries between empty and occupied seats.")
        preferred_side_weight = st.sidebar.number_input("Preferred Side Neighbour Weight", 
                                                        value=settings["weights"].get("preferred_side_weight", 1.0),
                                                        step=0.1, format="%.1f",
                                                        key='preferred_side_weight',
                                                        help="Penalty weight if a preferred side neighbour is missing.")
    else:
        side_weight = st.sidebar.number_input("Side Neighbour", value=5.0, step=1.0, format="%.1f",
                                              key='side_weight',
                                              help="Weight for repeated side neighbours.")
        front_weight = st.sidebar.number_input("Front Neighbour", value=2.0, step=1.0, format="%.1f",
                                               key='front_weight',
                                               help="Weight for repeated front neighbours.")
        diagonal_weight = st.sidebar.number_input("Diagonal Neighbour", value=1.0, step=1.0, format="%.1f",
                                                  key='diagonal_weight',
                                                  help="Weight for repeated diagonal neighbours.")
        corner_weight = st.sidebar.number_input("Corner Weight", value=3.0, step=0.1, format="%.1f",
                                                key='corner_weight',
                                                help="Weight for repeated corner seatings.")
        gender_weight = st.sidebar.number_input("Gender Weight", value=5.0, step=0.1, format="%.1f",
                                                key='gender_weight',
                                                help="Weight for adjacent same-gender seats.")
        fixed_weight = st.sidebar.number_input("Fixed Seat Diversity Weight", value=2.0, step=0.1, format="%.1f",
                                               key='fixed_weight',
                                               help="Extra weight for fixed-seat persons.")
        empty_weight = st.sidebar.number_input("Empty Seat Clustering Weight", value=5.0, step=0.1, format="%.1f",
                                               key='empty_weight',
                                               help="Weight for boundaries between empty and occupied seats.")
        preferred_side_weight = st.sidebar.number_input("Preferred Side Neighbour Weight", value=1.0, step=0.1, format="%.1f",
                                                        key='preferred_side_weight',
                                                        help="Penalty weight if a preferred side neighbour is missing.")
    
    with st.sidebar.expander("Optimization Parameters", expanded=False):
        if "uploaded_settings" in st.session_state:
            settings = st.session_state.uploaded_settings
            iterations = st.number_input("Iterations", 
                                         value=settings["optimization_params"]["iterations"], 
                                         step=1000, min_value=1000,
                                         key='iterations',
                                         help="Number of iterations for simulated annealing.")
            initial_temp = st.number_input("Initial Temperature", 
                                           value=settings["optimization_params"]["initial_temp"], 
                                           step=1.0,
                                           key='initial_temp',
                                           help="Starting temperature for simulated annealing.")
            cooling_rate = st.slider("Cooling Rate", min_value=0.990, max_value=0.9999, 
                                     value=settings["optimization_params"]["cooling_rate"],
                                     key='cooling_rate',
                                     help="Cooling multiplier per iteration.")
        else:
            iterations = st.number_input("Iterations", value=20000, step=1000, min_value=1000,
                                         key='iterations',
                                         help="Number of iterations for simulated annealing.")
            initial_temp = st.number_input("Initial Temperature", value=10.0, step=1.0,
                                           key='initial_temp',
                                           help="Starting temperature for simulated annealing.")
            cooling_rate = st.slider("Cooling Rate", min_value=0.990, max_value=0.9999, value=0.9995,
                                     key='cooling_rate',
                                     help="Cooling multiplier per iteration.")
    
    run_button = st.sidebar.button("Run Optimization")
    
    if run_button or "best_assignments" not in st.session_state:
        with st.spinner("Running optimization..."):
            best_assignments, best_cost, neighbors_info, corner_count, cost_history = run_optimization_and_build_data(
                iterations, initial_temp, cooling_rate,
                side_weight, front_weight, diagonal_weight,
                corner_weight, gender_weight, fixed_weight, empty_weight,
                people, person_genders, fixed_positions, TABLES,
                preferred_side_preferences, preferred_side_weight
            )
            st.session_state.best_assignments = best_assignments
            st.session_state.best_cost = best_cost
            st.session_state.neighbors_info = neighbors_info
            st.session_state.corner_count = corner_count
            st.session_state.person_genders = person_genders
            st.session_state.cost_history = cost_history
            
            combined_df = combine_all_seating_dataframes(best_assignments, TABLES, TABLE_LETTERS)
            st.session_state.combined_df = combined_df
            
            # Compute per-person cost breakdown using the new function.
            weights = {
                "side_neighbour_weight": side_weight,
                "front_neighbour_weight": front_weight,
                "diagonal_neighbour_weight": diagonal_weight,
                "corner_weight": corner_weight,
                "gender_weight": gender_weight,
                "fixed_weight": fixed_weight,
                "empty_weight": empty_weight,
                "preferred_side_weight": preferred_side_weight
            }
            indiv_costs = compute_individual_cost_breakdown(best_assignments, compute_seat_neighbors(TABLES), TABLES,
                                                              person_genders, fixed_positions, preferred_side_preferences,
                                                              weights)
            st.session_state.indiv_costs = indiv_costs
    
    st.success(f"Optimization complete. Best cost: {st.session_state.best_cost}")
    
    st.header("Seating Arrangements")
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
    for person, types_dict in st.session_state.neighbors_info.items():
        data.append({
            "Person": person,
            "Gender": st.session_state.person_genders.get(person, "X"),
            "Side Neighbours": ", ".join(sorted(types_dict["side"])),
            "Front Neighbours": ", ".join(sorted(types_dict["front"])),
            "Diagonal Neighbours": ", ".join(sorted(types_dict["diagonal"])),
            "Corner Count": st.session_state.corner_count.get(person, 0)
        })
    nbr_df = pd.DataFrame(data)
    st.dataframe(nbr_df, height=400)
    
    st.header("Cost Over Iterations")
    # Build a DataFrame from cost_history (each record contains iteration and cost components)
    cost_hist_df = pd.DataFrame(st.session_state.cost_history)
    cost_hist_df = cost_hist_df.set_index("iteration")
    st.line_chart(cost_hist_df[["total_cost", "neighbor_cost", "corner_cost", "gender_cost", "empty_cost", "preferred_side_cost"]])
    
    st.header("Individual Cost Breakdown")
    # Build a table from the individual cost breakdown
    indiv_data = []
    for person, comp in st.session_state.indiv_costs.items():
        indiv_data.append({
            "Person": person,
            "Side Cost": comp["side_cost"],
            "Front Cost": comp["front_cost"],
            "Diagonal Cost": comp["diagonal_cost"],
            "Neighbor Cost": comp["neighbor_cost"],
            "Corner Cost": comp["corner_cost"],
            "Preferred Side Cost": comp["preferred_side_cost"],
            "Gender Cost": comp["gender_cost"],
            "Empty Cost": comp["empty_cost"],
            "Total Cost": comp["total_cost"]
        })
    indiv_df = pd.DataFrame(indiv_data)
    st.dataframe(indiv_df, height=400)

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
