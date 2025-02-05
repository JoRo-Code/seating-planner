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
# 0. Default Values                 #
#####################################

# Table Defaults
DEFAULT_TABLE_DEF = "A: 3\nB: 3\nC: 4"

# Guest List Defaults
DEFAULT_MALE_NAMES = """John
Mike
David
Steve
Robert
James
William
Richard
Jonas
Thomas"""

DEFAULT_FEMALE_NAMES = """Mary
Linda
Susan
Karen
Patricia
Barbara
Nancy
Lisa
Betty
Margaret"""

# Assignment Defaults
DEFAULT_FIXED_SEATS = """John: A2
Mary: B2"""

DEFAULT_PREFERRED_SIDE = """John: Linda, Karen"""
DEFAULT_SPECIAL_COSTS = """John: 100"""

# Weight Defaults
DEFAULT_SIDE_WEIGHT = 5.0
DEFAULT_FRONT_WEIGHT = 2.0
DEFAULT_DIAGONAL_WEIGHT = 1.0
DEFAULT_CORNER_WEIGHT = 5.0
DEFAULT_GENDER_WEIGHT = 5.0
DEFAULT_FIXED_WEIGHT = 2.0
DEFAULT_EMPTY_WEIGHT = 5.0
DEFAULT_PREFERRED_SIDE_WEIGHT = 1.0
DEFAULT_UNIFORMITY_WEIGHT = 1.0

# Optimization Defaults
DEFAULT_ITERATIONS = 20000
DEFAULT_INITIAL_TEMP = 10.0
DEFAULT_COOLING_RATE = 0.9995
DEFAULT_NUM_ROUNDS = 3

#####################################
# 1. Table Definitions & Utilities  #
#####################################

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

def parse_special_cost_multipliers(text):
    multipliers = {}
    lines = text.strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        try:
            person_part, mult_str = line.split(":", 1)
            person = person_part.strip()
            multiplier = float(mult_str.strip())
            multipliers[person] = multiplier
        except Exception:
            continue
    return multipliers

#####################################
# 3. Cost & Neighbor Preference      #
#####################################

def compute_cost(assignments, seat_neighbors, tables, person_genders, fixed_positions,
                 side_weight=1.0, front_weight=1.0, diagonal_weight=1.0,
                 corner_weight=5.0,  # exponential base for corner penalty
                 gender_weight=5.0, fixed_weight=2.0, empty_weight=5.0,
                 preferred_side_preferences=None, preferred_side_weight=1.0,
                 uniformity_weight=1.0, special_cost_multipliers=None, breakdown=False):
    """
    Computes the overall cost over all seating rounds.
    
    Cost components:
      1. Neighbor cost: repeated neighbors.
      2. Corner cost (exponential): For each guest, if they sit on a corner c times,
         penalty = sum_{i=1}^{c} (corner_weight^i)
         e.g. if corner_weight is 5: 1 corner → 5; 2 corners → 5+25=30; 3 corners → 5+25+125=155.
      3. Gender cost: adjacent same-gender pairs.
      4. Empty seat clustering cost.
      5. Preferred side neighbour cost.
      6. Uniformity penalty: uniformity_weight times the variance of individual adjusted costs.
      7. Special cost multipliers: each guest's cost is multiplied by a specified factor.
    
    The overall cost is computed as the sum of the adjusted individual costs plus the uniformity penalty.
    
    If breakdown is True, returns (total_cost, breakdown_dict).
    """
    if special_cost_multipliers is None:
        special_cost_multipliers = {}
        
    # Compute individual cost breakdown (without special multiplier)
    indiv = compute_individual_cost_breakdown(assignments, seat_neighbors, tables,
                                               person_genders, fixed_positions,
                                               preferred_side_preferences,
                                               {"side_neighbour_weight": side_weight,
                                                "front_neighbour_weight": front_weight,
                                                "diagonal_neighbour_weight": diagonal_weight,
                                                "corner_weight": corner_weight,
                                                "gender_weight": gender_weight,
                                                "fixed_weight": fixed_weight,
                                                "empty_weight": empty_weight,
                                                "preferred_side_weight": preferred_side_weight})
    # Adjust each individual's cost by the special multiplier (default=1.0)
    adjusted_costs = {}
    regular_costs = {}  # Track costs for persons without special multipliers
    for person in indiv:
        multiplier = special_cost_multipliers.get(person, 1.0)
        cost = indiv[person]["total_cost"] * multiplier
        adjusted_costs[person] = cost
        if person not in special_cost_multipliers:
            regular_costs[person] = cost

    overall_indiv_cost = sum(adjusted_costs.values())
    
    # Calculate uniformity penalty based on the range and standard deviation
    if regular_costs:
        costs = list(regular_costs.values())
        avg_cost = sum(costs) / len(costs)
        
        # Calculate penalties for deviations from average
        squared_deviations = [(c - avg_cost) ** 2 for c in costs]
        max_deviation = max(abs(c - avg_cost) for c in costs)
        
        # Use exponential penalty for large deviations
        exp_penalty = sum(math.exp(abs(c - avg_cost) / avg_cost) for c in costs)
        
        uniformity_penalty = uniformity_weight * (sum(squared_deviations) + max_deviation + exp_penalty)
    else:
        uniformity_penalty = 0

    total_cost = overall_indiv_cost + uniformity_penalty

    if breakdown:
        breakdown_dict = {
            "overall_indiv_cost": overall_indiv_cost,
            "uniformity_cost": uniformity_penalty,
            "total_cost": total_cost,
            "adjusted_individual_costs": adjusted_costs
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
                    if n_seat in round_assign:
                        neighbor_person = round_assign[n_seat]
                        person_neighbors[person][n_type].add(neighbor_person)
    return person_neighbors, corner_count


#####################################
# 4. Optimization Functions          #
#####################################

def initialize_assignments_alternating(people, person_genders, tables, fixed_positions, num_rounds=3):
    """
    Creates initial seating assignments that try to alternate genders on every table row.
    It respects fixed seat assignments and, if available, uses free male and female lists.
    """
    seats = generate_seats(tables)
    assignments = []
    # Determine free persons for each gender after fixed assignments
    fixed_names = set(fixed_positions.keys())
    free_males = [p for p in people if person_genders.get(p) == "M" and p not in fixed_names]
    free_females = [p for p in people if person_genders.get(p) == "F" and p not in fixed_names]
    # Also maintain a combined list in case numbers are unbalanced
    free_all = [p for p in people if p not in fixed_names]

    for _ in range(num_rounds):
        round_assignment = {}
        # First, assign the fixed seats
        for person, seat in fixed_positions.items():
            round_assignment[seat] = person

        # Determine the free seats
        free_seats = [s for s in seats if s not in round_assignment]
        # Group free seats by table and row
        seats_by_table_row = {}
        for seat in free_seats:
            t, row, col = seat
            seats_by_table_row.setdefault((t, row), []).append(seat)

        # For each group (i.e. each row on a table), sort by column and assign alternating genders
        for (t, row), seat_list in seats_by_table_row.items():
            seat_list.sort(key=lambda s: s[2])
            for i, seat in enumerate(seat_list):
                # For even positions, try to assign a male; for odd, a female.
                if i % 2 == 0:
                    if free_males:
                        person = free_males.pop(0)
                    elif free_all:
                        person = free_all.pop(0)
                    else:
                        person = "Empty"
                else:
                    if free_females:
                        person = free_females.pop(0)
                    elif free_all:
                        person = free_all.pop(0)
                    else:
                        person = "Empty"
                round_assignment[seat] = person

        # If any free persons remain (because the count wasn’t exactly equal) assign them arbitrarily.
        remaining_seats = [s for s in free_seats if s not in round_assignment]
        for seat in remaining_seats:
            if free_all:
                round_assignment[seat] = free_all.pop(0)
            else:
                round_assignment[seat] = "Empty"
        assignments.append(round_assignment)
    return assignments

def initialize_assignments_alternating(people, person_genders, tables, fixed_positions, num_rounds=3):
    """
    Creates initial seating assignments that try to alternate genders on every table row,
    while taking into account fixed positions. It assumes the fixed positions already
    follow an alternating pattern so that an alternating arrangement is possible.
    
    Parameters:
      - people: list of all guest names.
      - person_genders: dict mapping name -> "M" or "F".
      - tables: dict mapping table ID -> seats per side.
      - fixed_positions: dict mapping guest name -> seat (tuple, e.g. (table, row, col)).
      - num_rounds: number of seating arrangements (rounds) to generate.
      
    Returns:
      - assignments: list (length num_rounds) of dicts mapping seat -> person.
    """
    # Generate all seat coordinates
    seats = generate_seats(tables)
    assignments = []
    
    # Build a mapping from seat -> fixed person (for easier lookup)
    fixed_seat_assignments = {seat: person for person, seat in fixed_positions.items()}
    fixed_names = set(fixed_positions.keys())
    
    for _ in range(num_rounds):
        round_assignment = {}
        # First, assign fixed positions
        round_assignment.update(fixed_seat_assignments)
        
        # Group free seats by (table, row)
        free_seats = [s for s in seats if s not in round_assignment]
        seats_by_table_row = {}
        for seat in free_seats:
            t, row, col = seat
            seats_by_table_row.setdefault((t, row), []).append(seat)
        
        # Create working lists for free persons of each gender
        free_males = [p for p in people if person_genders.get(p) == "M" and p not in fixed_names]
        free_females = [p for p in people if person_genders.get(p) == "F" and p not in fixed_names]
        # Also maintain a combined free list for fallback
        free_all = [p for p in people if p not in fixed_names]
        
        # Process each table row
        for (t, row), seat_list in seats_by_table_row.items():
            # Sort the seats by column order
            seat_list.sort(key=lambda s: s[2])
            
            # Check if any fixed seat exists in this table row.
            # We'll build a list of (seat, fixed_person) for seats in this table and row.
            fixed_in_row = [(seat, fixed_seat_assignments[seat]) 
                            for seat in fixed_seat_assignments if seat[0] == t and seat[1] == row]
            
            if fixed_in_row:
                # Use the leftmost fixed seat as the reference.
                fixed_in_row.sort(key=lambda x: x[0][2])  # sort by column
                ref_seat, ref_person = fixed_in_row[0]
                ref_col = ref_seat[2]
                ref_gender = person_genders.get(ref_person, "X")
                def desired_gender(seat):
                    _, _, col = seat
                    # If col differs from the reference by an even number, use the same gender
                    if (col - ref_col) % 2 == 0:
                        return ref_gender
                    else:
                        return "F" if ref_gender == "M" else "M"
            # If there is no fixed seat in the row, use a default alternating pattern.
            # (Here we assign based on the order in the sorted list: even-index = "M", odd-index = "F".)
            
            for idx, seat in enumerate(seat_list):
                if fixed_in_row:
                    d_gender = desired_gender(seat)
                else:
                    d_gender = "M" if idx % 2 == 0 else "F"
                    
                # Attempt to assign a free person of the desired gender.
                if d_gender == "M" and free_males:
                    person = free_males.pop(0)
                    if person in free_all:
                        free_all.remove(person)
                elif d_gender == "F" and free_females:
                    person = free_females.pop(0)
                    if person in free_all:
                        free_all.remove(person)
                else:
                    # Fallback: assign from any free person
                    if free_all:
                        person = free_all.pop(0)
                        if person in free_males:
                            free_males.remove(person)
                        if person in free_females:
                            free_females.remove(person)
                    else:
                        person = "Empty"
                round_assignment[seat] = person
        
        # (For safety, assign any seats not yet assigned—should not happen if all seats were processed.)
        for seat in seats:
            if seat not in round_assignment:
                round_assignment[seat] = free_all.pop(0) if free_all else "Empty"
                    
        assignments.append(round_assignment)
    return assignments



def initialize_assignments(people, tables, fixed_positions, num_rounds=3):
    seats = generate_seats(tables)
    #free_people = set(people)
    free_people = list(people)
    for person in fixed_positions:
            if person in free_people:       
                free_people.remove(person)
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
                         corner_weight=5.0,  # exponential base for corner penalty
                         gender_weight=5.0, fixed_weight=2.0, empty_weight=5.0,
                         preferred_side_preferences=None, preferred_side_weight=1.0,
                         uniformity_weight=1.0, special_cost_multipliers=None,
                         record_history=True):
    """
    Uses simulated annealing to optimize seating assignments.
    Now, in addition to swapping two free seats in a round, this function also occasionally
    attempts a flip move: it picks a table in a round and flips a contiguous block of free seats 
    (from a corner) horizontally. The idea is to correct, for example, a same‐gender pairing in the middle.
    
    Parameters:
      ... (same as before)
      special_cost_multipliers: dict mapping person to a multiplier (optional)
      record_history: if True, record the cost history
      
    Returns:
      best_assignments, best_cost, cost_history
    """
    if special_cost_multipliers is None:
        special_cost_multipliers = {}
    if record_history:
        current_cost, current_breakdown = compute_cost(
            assignments, seat_neighbors, tables, person_genders, fixed_positions,
            side_weight, front_weight, diagonal_weight, corner_weight,
            gender_weight, fixed_weight, empty_weight,
            preferred_side_preferences, preferred_side_weight,
            uniformity_weight, special_cost_multipliers, breakdown=True
        )
    else:
        current_cost = compute_cost(
            assignments, seat_neighbors, tables, person_genders, fixed_positions,
            side_weight, front_weight, diagonal_weight, corner_weight,
            gender_weight, fixed_weight, empty_weight,
            preferred_side_preferences, preferred_side_weight,
            uniformity_weight, special_cost_multipliers, breakdown=False
        )
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
    
    # New parameter: chance to try a flip move instead of a swap move.
    flip_probability = 0.2  # 20% chance; adjust as desired

    for iter_num in range(1, iterations + 1):
        # Decide whether to try a flip move or the usual swap move.
        if random.random() < flip_probability:
            # ------------------ FLIP MOVE ------------------
            # Choose a round at random.
            r = random.randint(0, num_rounds - 1)
            # Choose a random table.
            table_id = random.choice(list(tables.keys()))
            seats_per_side = tables[table_id]
            # Get free seats (i.e. non-fixed seats) for this table in round r.
            free_seats_table = [s for s in free_seats_by_round[r] if s[0] == table_id]
            # Choose a corner at random: "left" or "right"
            corner = random.choice(["left", "right"])
            
            # Define a helper to compute the maximum contiguous block width from a given corner
            def max_block_width_for_row(row, corner):
                width = 0
                if corner == "left":
                    for col in range(seats_per_side):
                        if (table_id, row, col) in free_seats_table:
                            width += 1
                        else:
                            break
                else:  # "right" corner: start at the rightmost column and go left
                    for col in range(seats_per_side - 1, -1, -1):
                        if (table_id, row, col) in free_seats_table:
                            width += 1
                        else:
                            break
                return width

            max_width_row0 = max_block_width_for_row(0, corner)
            max_width_row1 = max_block_width_for_row(1, corner)
            max_width = min(max_width_row0, max_width_row1)
            # If no contiguous free block exists, skip the flip move.
            if max_width < 1:
                move_type = "swap"
            else:
                move_type = "flip"
        else:
            move_type = "swap"
            
        if move_type == "swap":
            # ------------------ SWAP MOVE (existing move) ------------------
            r = random.randint(0, num_rounds - 1)
            seat1, seat2 = random.sample(free_seats_by_round[r], 2)
            assignments[r][seat1], assignments[r][seat2] = assignments[r][seat2], assignments[r][seat1]
            
            if record_history:
                new_cost, new_breakdown = compute_cost(
                    assignments, seat_neighbors, tables, person_genders, fixed_positions,
                    side_weight, front_weight, diagonal_weight, corner_weight,
                    gender_weight, fixed_weight, empty_weight,
                    preferred_side_preferences, preferred_side_weight,
                    uniformity_weight, special_cost_multipliers, breakdown=True
                )
            else:
                new_cost = compute_cost(
                    assignments, seat_neighbors, tables, person_genders, fixed_positions,
                    side_weight, front_weight, diagonal_weight, corner_weight,
                    gender_weight, fixed_weight, empty_weight,
                    preferred_side_preferences, preferred_side_weight,
                    uniformity_weight, special_cost_multipliers, breakdown=False
                )
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_assignments = copy.deepcopy(assignments)
                if record_history:
                    current_breakdown = new_breakdown
            else:
                # revert the swap
                assignments[r][seat1], assignments[r][seat2] = assignments[r][seat2], assignments[r][seat1]
        elif move_type == "flip":
            # ------------------ FLIP MOVE ------------------
            # We already chose r, table_id, corner, and computed max_width.
            # Choose a block width k (at least 1) randomly from the available range.
            k = random.randint(1, max_width)
            # Identify the block of seats to flip.
            # (Remember: each table has two rows.)
            if corner == "left":
                block_seats = [(table_id, row, col) for row in [0, 1] for col in range(k)]
            else:  # right corner
                block_seats = [(table_id, row, col) for row in [0, 1] for col in range(seats_per_side - k, seats_per_side)]
            # Save the current occupants of these seats so that we can revert if needed.
            old_block = {seat: assignments[r][seat] for seat in block_seats}
            # Create a new assignment for round r by flipping each row’s block.
            new_assignment = assignments[r].copy()
            for row in [0, 1]:
                if corner == "left":
                    row_seats = [(table_id, row, col) for col in range(k)]
                else:
                    row_seats = [(table_id, row, col) for col in range(seats_per_side - k, seats_per_side)]
                reversed_row = list(reversed(row_seats))
                for original_seat, flipped_seat in zip(row_seats, reversed_row):
                    new_assignment[original_seat] = assignments[r][flipped_seat]
            assignments[r] = new_assignment
            
            if record_history:
                new_cost, new_breakdown = compute_cost(
                    assignments, seat_neighbors, tables, person_genders, fixed_positions,
                    side_weight, front_weight, diagonal_weight, corner_weight,
                    gender_weight, fixed_weight, empty_weight,
                    preferred_side_preferences, preferred_side_weight,
                    uniformity_weight, special_cost_multipliers, breakdown=True
                )
            else:
                new_cost = compute_cost(
                    assignments, seat_neighbors, tables, person_genders, fixed_positions,
                    side_weight, front_weight, diagonal_weight, corner_weight,
                    gender_weight, fixed_weight, empty_weight,
                    preferred_side_preferences, preferred_side_weight,
                    uniformity_weight, special_cost_multipliers, breakdown=False
                )
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_assignments = copy.deepcopy(assignments)
                if record_history:
                    current_breakdown = new_breakdown
            else:
                # Revert the flip move.
                for seat, occupant in old_block.items():
                    assignments[r][seat] = occupant

        temp *= cooling_rate

        if record_history and (iter_num % 100 == 0):
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
    Computes a cost breakdown per person.
    Returns a dict mapping person -> dict with keys:
      side_cost, front_cost, diagonal_cost, neighbor_cost,
      corner_cost, preferred_side_cost, gender_cost, empty_cost, total_cost.
    The corner_cost is computed exponentially: for count c,
         penalty = sum_{i=1}^{c} (corner_weight^i)
    """
    side_weight = weights.get("side_neighbour_weight", 1.0)
    front_weight = weights.get("front_neighbour_weight", 1.0)
    diagonal_weight = weights.get("diagonal_neighbour_weight", 1.0)
    corner_weight = weights.get("corner_weight", 5.0)
    gender_weight = weights.get("gender_weight", 5.0)
    fixed_weight = weights.get("fixed_weight", 2.0)
    empty_weight = weights.get("empty_weight", 5.0)
    preferred_side_weight = weights.get("preferred_side_weight", 1.0)
    
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
    # Compute neighbor costs per person.
    indiv_neighbors = defaultdict(lambda: {"side": [], "front": [], "diagonal": []})
    for round_assign in assignments:
        for seat, person in round_assign.items():
            for n_type, n_list in seat_neighbors[seat].items():
                for n_seat in n_list:
                    # Add check for valid neighbor seat
                    if n_seat in round_assign:
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

    # Corner cost per person.
    corner_count = defaultdict(int)
    for round_assign in assignments:
        for seat, person in round_assign.items():
            t, row, col = seat
            if col == 0 or col == tables[t] - 1:
                corner_count[person] += 1
    for person, count in corner_count.items():
        if person_genders.get(person, "X") == "X":
            continue
        penalty = 0
        for i in range(1, count + 1):
            penalty += corner_weight ** i
        indiv_breakdown[person]["corner_cost"] = penalty

    # Preferred side neighbour cost per person
    for person in preferred_side_preferences:
        if person_genders.get(person, "X") == "X":
            continue
        desired = set(preferred_side_preferences[person])
        # Track which desired neighbors we've found
        found_neighbors = set()
        for round_assign in assignments:
            for seat, seated_person in round_assign.items():
                if seated_person == person:
                    # Only consider valid neighbor seats that exist in the assignment
                    valid_neighbors = [n_seat for n_seat in seat_neighbors[seat]["side"] if n_seat in round_assign]
                    side_nbrs = set(round_assign[n_seat] for n_seat in valid_neighbors)
                    found_neighbors.update(desired & side_nbrs)
        # Count how many desired neighbors were never found as side neighbors
        missing = len(desired - found_neighbors)
        indiv_breakdown[person]["preferred_side_cost"] = missing * preferred_side_weight

    # Gender cost: for each adjacent same-gender pair, attribute half cost to each person.
    for round_assign in assignments:
        valid_seats = set(round_assign.keys())  # Only consider seats that exist in this round
        for t, seats_per_side in tables.items():
            for row in [0, 1]:
                seats_in_row = [(t, row, col) for col in range(seats_per_side)]
                seats_in_row = [s for s in seats_in_row if s in valid_seats]  # Filter to valid seats
                for i in range(len(seats_in_row) - 1):
                    s1, s2 = seats_in_row[i], seats_in_row[i+1]
                    p1, p2 = round_assign[s1], round_assign[s2]
                    g1 = person_genders.get(p1, "X")
                    g2 = person_genders.get(p2, "X")
                    if g1 == "X" or g2 == "X":
                        continue
                    if g1 == g2:
                        indiv_breakdown[p1]["gender_cost"] += gender_weight 
                        indiv_breakdown[p2]["gender_cost"] += gender_weight
    
    # Additional penalty for front neighbors (vertical pairing)
    # In a 2-row table, the “front” neighbor is the seat in the opposite row with the same column.
    for round_assign in assignments:
        for t, seats_per_side in tables.items():
            for col in range(seats_per_side):
                s0 = (t, 0, col)
                s1 = (t, 1, col)
                if s0 in round_assign and s1 in round_assign:
                    p0, p1 = round_assign[s0], round_assign[s1]
                    g0, g1 = person_genders.get(p0, "X"), person_genders.get(p1, "X")
                    if g0 != "X" and g1 != "X" and g0 == g1:
                        indiv_breakdown[p0]["gender_cost"] += gender_weight
                        indiv_breakdown[p1]["gender_cost"] += gender_weight

    # Empty cost: for each adjacent pair where one is empty and one is not, assign cost to the non-empty person.
    for round_assign in assignments:
        valid_seats = set(round_assign.keys())  # Only consider seats that exist in this round
        for t, seats_per_side in tables.items():
            for row in [0, 1]:
                seats_in_row = [(t, row, col) for col in range(seats_per_side)]
                seats_in_row = [s for s in seats_in_row if s in valid_seats]  # Filter to valid seats
                for i in range(len(seats_in_row) - 1):
                    s1, s2 = seats_in_row[i], seats_in_row[i+1]
                    p1, p2 = round_assign[s1], round_assign[s2]
                    if p1.startswith("Empty") and not p2.startswith("Empty"):
                        indiv_breakdown[p2]["empty_cost"] += empty_weight
                    elif p2.startswith("Empty") and not p1.startswith("Empty"):
                        indiv_breakdown[p1]["empty_cost"] += empty_weight



    # Total cost per person.
    for person, comp in indiv_breakdown.items():
        comp["total_cost"] = (comp["neighbor_cost"] +
                              comp["corner_cost"] +
                              comp["preferred_side_cost"] +
                              comp["gender_cost"] +
                              comp["empty_cost"])
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
            
            # Extract table letter and seat number.
            table_letter = seat_id[0]
            seat_num = int(seat_id[1:]) - 1  # 0-based index.
            
            # Find table_id from table letter.
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
                                      corner_weight,  # exponential base for corner penalty
                                      gender_weight, fixed_weight, empty_weight,
                                      people, person_genders, fixed_positions, tables,
                                      preferred_side_preferences, preferred_side_weight,
                                      uniformity_weight, special_cost_multipliers,
                                      num_rounds=3):
    """
    Runs the seating optimization and returns:
      - best_assignments: list (per round) of seat -> person assignments,
      - best_cost: final cost,
      - neighbors_info: dict mapping each person to neighbor info,
      - corner_count: dict mapping each person to corner counts,
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
    assignments = initialize_assignments_alternating(people, person_genders=person_genders, tables=tables, fixed_positions=fixed_positions, num_rounds=num_rounds)
    best_assignments, best_cost, cost_history = optimize_assignments(
        assignments, seat_neighbors, tables, fixed_positions, person_genders,
        iterations, initial_temp, cooling_rate,
        side_weight, front_weight, diagonal_weight, corner_weight,
        gender_weight, fixed_weight, empty_weight,
        preferred_side_preferences, preferred_side_weight,
        uniformity_weight, special_cost_multipliers,
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
    default_male = DEFAULT_MALE_NAMES
    default_female = DEFAULT_FEMALE_NAMES
    male_text = st.session_state.male_text if 'male_text' in st.session_state else default_male
    female_text = st.session_state.female_text if 'female_text' in st.session_state else default_female
    fixed_text = st.session_state.fixed_text if 'fixed_text' in st.session_state else DEFAULT_FIXED_SEATS
    pref_side_text = st.session_state.pref_side_text if 'pref_side_text' in st.session_state else DEFAULT_PREFERRED_SIDE
    special_cost_text = st.session_state.special_cost_multipliers_text if 'special_cost_multipliers_text' in st.session_state else DEFAULT_SPECIAL_COSTS
    iterations = st.session_state.iterations if 'iterations' in st.session_state else DEFAULT_ITERATIONS
    initial_temp = st.session_state.initial_temp if 'initial_temp' in st.session_state else DEFAULT_INITIAL_TEMP
    cooling_rate = st.session_state.cooling_rate if 'cooling_rate' in st.session_state else DEFAULT_COOLING_RATE
    side_weight = st.session_state.side_weight if 'side_weight' in st.session_state else DEFAULT_SIDE_WEIGHT
    front_weight = st.session_state.front_weight if 'front_weight' in st.session_state else DEFAULT_FRONT_WEIGHT
    diagonal_weight = st.session_state.diagonal_weight if 'diagonal_weight' in st.session_state else DEFAULT_DIAGONAL_WEIGHT
    corner_weight = st.session_state.corner_weight if 'corner_weight' in st.session_state else DEFAULT_CORNER_WEIGHT
    gender_weight = st.session_state.gender_weight if 'gender_weight' in st.session_state else DEFAULT_GENDER_WEIGHT
    fixed_weight = st.session_state.fixed_weight if 'fixed_weight' in st.session_state else DEFAULT_FIXED_WEIGHT
    empty_weight = st.session_state.empty_weight if 'empty_weight' in st.session_state else DEFAULT_EMPTY_WEIGHT
    preferred_side_weight = st.session_state.preferred_side_weight if 'preferred_side_weight' in st.session_state else DEFAULT_PREFERRED_SIDE_WEIGHT
    uniformity_weight = st.session_state.uniformity_weight if 'uniformity_weight' in st.session_state else DEFAULT_UNIFORMITY_WEIGHT
    special_cost_multipliers = parse_special_cost_multipliers(special_cost_text)
    num_rounds = st.session_state.num_rounds if 'num_rounds' in st.session_state else DEFAULT_NUM_ROUNDS
    
    return {
        "table_definitions": table_def_text,
        "male_names": male_text,
        "female_names": female_text,
        "fixed_assignments": fixed_text,
        "preferred_side_preferences_text": pref_side_text,
        "special_cost_multipliers_text": special_cost_text,
        "optimization_params": {
            "iterations": iterations,
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
            "num_rounds": num_rounds
        },
        "weights": {
            "side_neighbour_weight": side_weight,
            "front_neighbour_weight": front_weight,
            "diagonal_neighbour_weight": diagonal_weight,
            "corner_weight": corner_weight,
            "gender_weight": gender_weight,
            "fixed_weight": fixed_weight,
            "empty_weight": empty_weight,
            "preferred_side_weight": preferred_side_weight,
            "uniformity_weight": uniformity_weight
        }
    }
    
def generate_table_html_with_highlights(arrangement, table_id, table_letter, tables, highlights):
    """
    Generates an HTML representation of one table for one seating arrangement.
    - arrangement: dict mapping seat (table, row, col) to occupant name.
    - highlights: dict mapping seat tuples to a highlight category:
         "selected" for the chosen person,
         "side" for immediate (side) neighbours,
         "other" for front and diagonal neighbours.
    """
    num_cols = tables[table_id]
    # Define common cell style and highlight colors.
    cell_style = (
        "width:60px; height:60px; border:1px solid #000; display:flex; "
        "align-items:center; justify-content:center; margin:2px; font-size:12px; font-weight:bold;"
    )
    highlight_colors = {
        "selected": "#FFD700",  # gold
        "side": "#90EE90",      # light green
        "other": "#ADD8E6"      # light blue
    }
    
    def get_bg_color(row, col):
        seat = (table_id, row, col)
        if seat in highlights:
            return highlight_colors.get(highlights[seat], "#ffffff")
        else:
            # Default background is now white for all seats.
            return "#ffffff"
    
    # Build the two rows: top row (row 0) and bottom row (row 1).
    top_html = "<div style='display:flex; justify-content:center;'>"
    for col in range(num_cols):
        seat = (table_id, 0, col)
        occupant = arrangement.get(seat, "")
        bg_color = get_bg_color(0, col)
        top_html += f"<div style='{cell_style} background-color:{bg_color};'>{occupant}</div>"
    top_html += "</div>"
    
    bottom_html = "<div style='display:flex; justify-content:center;'>"
    for col in range(num_cols):
        seat = (table_id, 1, col)
        occupant = arrangement.get(seat, "")
        bg_color = get_bg_color(1, col)
        bottom_html += f"<div style='{cell_style} background-color:{bg_color};'>{occupant}</div>"
    bottom_html += "</div>"
    
    full_html = f"""
    <html>
      <head>
        <meta charset="UTF-8">
        <style>
          body {{ font-family: sans-serif; margin:10px; padding:0; }}
        </style>
      </head>
      <body>
        <h4 style="text-align:center;">Table {table_letter}</h4>
        {top_html}
        {bottom_html}
      </body>
    </html>
    """
    return full_html


def display_highlighted_arrangements_by_names(selected_person, best_assignments, tables, table_letters, aggregated_neighbors):
    """
    Displays each seating arrangement by highlighting seats based on occupant names.
    """
    # --- Display a Legend with styled boxes ---
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h4 style="margin-bottom: 10px;"></h4>
        <div style="display: flex; gap: 15px; flex-wrap: wrap;">
            <div style="display: flex; align-items: center;">
                <div style="width: 30px; height: 30px; background-color: #FFD700; border: 1px solid #000; margin-right: 8px;"></div>
                <span>Selected Person</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 30px; height: 30px; background-color: #90EE90; border: 1px solid #000; margin-right: 8px;"></div>
                <span>Immediate (Side) Neighbour</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 30px; height: 30px; background-color: #ADD8E6; border: 1px solid #000; margin-right: 8px;"></div>
                <span>Front/Diagonal Neighbour</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Combine front and diagonal into a single set for "other" neighbors.
    other_neighbors = aggregated_neighbors.get("front", set()) | aggregated_neighbors.get("diagonal", set())
    
    # Loop over each arrangement.
    for round_index, arrangement in enumerate(best_assignments):
        st.subheader(f"Arrangement {round_index+1}")
        for table_id in sorted(tables.keys()):
            table_letter = table_letters[table_id]
            highlights = {}
            # List all seats for this table.
            seats_in_table = [(table_id, row, col) for row in [0, 1] for col in range(tables[table_id])]
            for seat in seats_in_table:
                occupant = arrangement.get(seat, "")
                if occupant == selected_person:
                    highlights[seat] = "selected"
                elif occupant in aggregated_neighbors.get("side", set()):
                    highlights[seat] = "side"
                elif occupant in other_neighbors:
                    highlights[seat] = "other"
            st.markdown(f"**Table {table_letter}**")
            html = generate_table_html_with_highlights(arrangement, table_id, table_letter, tables, highlights)
            components.html(html, height=180, scrolling=False)


def display_highlighted_arrangements(selected_person, best_assignments, tables, table_letters, seat_neighbors):
    """
    For the selected_person, aggregates neighbor information from all arrangements,
    then displays each seating arrangement with:
      - All seats where the person sat (in any arrangement) highlighted as "selected"
      - Any seat that was an immediate (side) neighbor in any round highlighted as "side"
      - Any seat that was a front or diagonal neighbor in any round highlighted as "other"
    Also displays a legend for the color codes.
    """
    # --- Aggregate the neighbor info across all arrangements ---
    global_selected_seats = set()
    global_side_neighbors = set()
    global_other_neighbors = set()
    
    for arrangement in best_assignments:
        for seat, occupant in arrangement.items():
            if occupant == selected_person:
                global_selected_seats.add(seat)
                # Add immediate (side) neighbors:
                for nbr in seat_neighbors[seat]["side"]:
                    global_side_neighbors.add(nbr)
                # Add front and diagonal neighbors:
                for nbr in seat_neighbors[seat]["front"] + seat_neighbors[seat]["diagonal"]:
                    global_other_neighbors.add(nbr)
    
    # Remove any overlap: a seat that is in global_selected_seats should not also be in neighbor sets.
    global_side_neighbors -= global_selected_seats
    global_other_neighbors -= (global_selected_seats | global_side_neighbors)
    
    # --- Display a Legend ---
    st.markdown("""
    **Legend:**  
    <span style="background-color: #FFD700; padding: 4px 8px; border: 1px solid #000; margin-right: 8px;">Selected Person<br>(where they sat in any arrangement)</span>  
    <span style="background-color: #90EE90; padding: 4px 8px; border: 1px solid #000; margin-right: 8px;">Immediate (Side) Neighbour<br>(aggregated)</span>  
    <span style="background-color: #ADD8E6; padding: 4px 8px; border: 1px solid #000;">Front/Diagonal Neighbour<br>(aggregated)</span>
    """, unsafe_allow_html=True)
    
    st.header(f"Aggregated Highlights for **{selected_person}**")
    
    # --- Loop over each arrangement ---
    for round_index, arrangement in enumerate(best_assignments):
        st.subheader(f"Arrangement {round_index+1}")
        for table_id in sorted(tables.keys()):
            table_letter = table_letters[table_id]
            highlights = {}
            # List all seats in this table.
            seats_in_table = [(table_id, row, col) for row in [0, 1] for col in range(tables[table_id])]
            for seat in seats_in_table:
                if seat in global_selected_seats:
                    highlights[seat] = "selected"
                elif seat in global_side_neighbors:
                    highlights[seat] = "side"
                elif seat in global_other_neighbors:
                    highlights[seat] = "other"
            st.markdown(f"**Table {table_letter}**")
            html = generate_table_html_with_highlights(arrangement, table_id, table_letter, tables, highlights)
            components.html(html, height=180, scrolling=False)


def main():
    st.title("SeatPlan")
    st.markdown("##### Optimizing seating arrangements for events.")
    st.markdown("###### Use the sidebar for additional customization")
    
    # Sidebar: Settings download/upload
    st.sidebar.markdown("# Settings")

    
    # Table Layout Configuration
    with st.expander("Table Configurations"):
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
        
        run_button = st.sidebar.button("Run Optimization", type="primary")
        
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
            num_rounds = st.number_input("Number of Rounds", 
                                         value=settings["optimization_params"].get("num_rounds", DEFAULT_NUM_ROUNDS),
                                         step=1, min_value=1,
                                         key='num_rounds',
                                         help="Number of seating arrangements to generate.")
        else:
            iterations = st.number_input("Iterations", value=DEFAULT_ITERATIONS, step=1000, min_value=1000,
                                         key='iterations',
                                         help="Number of iterations for simulated annealing.")
            initial_temp = st.number_input("Initial Temperature", value=DEFAULT_INITIAL_TEMP, step=1.0,
                                           key='initial_temp',
                                           help="Starting temperature for simulated annealing.")
            cooling_rate = st.slider("Cooling Rate", min_value=0.990, max_value=0.9999, value=DEFAULT_COOLING_RATE,
                                     key='cooling_rate',
                                     help="Cooling multiplier per iteration.")
            num_rounds = st.number_input("Number of Rounds", 
                                         value=DEFAULT_NUM_ROUNDS,
                                         step=1, min_value=1,
                                         key='num_rounds',
                                         help="Number of seating arrangements to generate.")
    
    
    # Guests
    with st.sidebar.expander("Guests"):
        st.markdown("""
            Enter the names of the guests.
        """)
        if "uploaded_settings" in st.session_state:
            settings = st.session_state.uploaded_settings
            male_text = st.text_area("Males (one per line)", 
                                            value=settings["male_names"], height=150,
                                            key='male_text', help="One male name per line.")
            male_count = len([name for name in male_text.splitlines() if name.strip()])
            st.caption(f"Male count: {male_count}")
            
            female_text = st.text_area("Females (one per line)", 
                                            value=settings["female_names"], height=150,
                                            key='female_text', help="One female name per line.")
            female_count = len([name for name in female_text.splitlines() if name.strip()])
            st.caption(f"Female count: {female_count}")
        else:
            default_male = DEFAULT_MALE_NAMES
            default_female = DEFAULT_FEMALE_NAMES
            male_text = st.text_area("Males (one per line)", value=default_male, height=150,
                                            key='male_text', help="One male name per line.")
            male_count = len([name for name in male_text.splitlines() if name.strip()])
            st.caption(f"Male count: {male_count}")
            
            female_text = st.text_area("Females (one per line)", value=default_female, height=150,
                                            key='female_text', help="One female name per line.")
            female_count = len([name for name in female_text.splitlines() if name.strip()])
            st.caption(f"Female count: {female_count}")
        male_names = [name.strip() for name in male_text.splitlines() if name.strip()]
        female_names = [name.strip() for name in female_text.splitlines() if name.strip()]
        people = male_names + female_names
        person_genders = {}
        for name in male_names:
            person_genders[name] = "M"
        for name in female_names:
            person_genders[name] = "F"
    
    # Fixed seat assignments
    with st.sidebar.expander("Special Guest Preferences"):
        st.markdown("""
            Some guests may be treated differently.
        """)
        st.header("Fixed Seat Assignments")
        if "uploaded_settings" in st.session_state:
            settings = st.session_state.uploaded_settings
            fixed_text = st.text_area("", 
                                            value=settings["fixed_assignments"], height=100,
                                            key='fixed_text', help="These guests will have assigned seats (e.g., 'John: A12'))")
        else:
            fixed_text = st.text_area("",
                                            value="John: A2\nMary: B2", height=100,
                                            key='fixed_text', help="These guests will have assigned seats (e.g., 'John: A12')")
        fixed_positions = parse_fixed_seats(fixed_text)
        
        # Preferred side neighbour preferences
        st.header("Preferred Side Neighbours")
        pref_side_text = st.text_area(
            "", 
            value=DEFAULT_PREFERRED_SIDE,  # Set default value here
            height=100,
            key='pref_side_text',
            help="For example: Alice: Bob, Charlie. Meaning Alice prefers to sit next to Bob and Charlie. Each line is one person and their preferred neighbours."
        )
        preferred_side_preferences = parse_preferred_side_neighbours(pref_side_text)
        
        # Special Cost Multipliers
        st.header("Special Cost Multipliers")
        special_cost_text = st.text_area(
            "Example: Alice: 0.5  (means Alice's penalty is halved)", 
            value=DEFAULT_SPECIAL_COSTS,  # Set default value here
            height=100,
            key='special_cost_multipliers_text',
            help="For guests whose cost you want to lower (or raise), enter one per line in the format: **Name: multiplier**. (A multiplier less than 1 lowers the cost.)"
        )
        special_cost_multipliers = parse_special_cost_multipliers(special_cost_text)
        
    # Conditions & Weights
    with st.sidebar.expander("Conditions & Weights", expanded=False):
        st.markdown("""
            Set the importance of each condition. Higher values make the condition more important.
        """)
        if "uploaded_settings" in st.session_state:
            settings = st.session_state.uploaded_settings
            side_weight = st.number_input("Side Neighbour", 
                                          value=settings["weights"].get("side_neighbour_weight", DEFAULT_SIDE_WEIGHT),
                                          step=1.0, format="%.1f",
                                          key='side_weight',
                                          help="Weight for repeated side neighbours.")
            front_weight = st.number_input("Front Neighbour", 
                                               value=settings["weights"].get("front_neighbour_weight", DEFAULT_FRONT_WEIGHT),
                                               step=1.0, format="%.1f",
                                               key='front_weight',
                                               help="Weight for repeated front neighbours.")
            diagonal_weight = st.number_input("Diagonal Neighbour", 
                                                  value=settings["weights"].get("diagonal_neighbour_weight", DEFAULT_DIAGONAL_WEIGHT),
                                                  step=1.0, format="%.1f",
                                                  key='diagonal_weight',
                                                  help="Weight for repeated diagonal neighbours.")
            corner_weight = st.number_input("Corner", 
                                                value=settings["weights"]["corner_weight"], 
                                                step=0.1, format="%.1f",
                                                key='corner_weight',
                                                help="Exponential base for corner penalty. For example, if 5 then first corner costs 5, second 25, third 125. Thus if two corners, the cost is 5+25=30.")
            gender_weight = st.number_input("Gender", 
                                                value=settings["weights"]["gender_weight"], 
                                                step=0.1, format="%.1f",
                                                key='gender_weight',
                                                help="Weight for adjacent same-gender seats.")
            fixed_weight = st.number_input("Fixed Seat Diversity", 
                                               value=settings["weights"]["fixed_weight"], 
                                               step=0.1, format="%.1f",
                                               key='fixed_weight',
                                               help="Extra weight for fixed-seat persons.")
            empty_weight = st.number_input("Empty Seat Clustering", 
                                               value=settings["weights"]["empty_weight"], 
                                               step=0.1, format="%.1f",
                                               key='empty_weight',
                                               help="Weight for boundaries between empty and occupied seats.")
            preferred_side_weight = st.number_input("Preferred Side Neighbour", 
                                                        value=settings["weights"].get("preferred_side_weight", DEFAULT_PREFERRED_SIDE_WEIGHT),
                                                        step=0.1, format="%.1f",
                                                        key='preferred_side_weight',
                                                        help="Penalty weight if a preferred side neighbour is missing.")
            uniformity_weight = st.number_input("Uniformity", 
                                                value=settings["weights"].get("uniformity_weight", DEFAULT_UNIFORMITY_WEIGHT),
                                                step=0.1, format="%.1f",
                                                key='uniformity_weight',
                                                help="Extra penalty for uneven distribution of individual costs.")
        else:
            side_weight = st.number_input("Side Neighbour", value=DEFAULT_SIDE_WEIGHT, step=1.0, format="%.1f",
                                          key='side_weight',
                                          help="Weight for repeated side neighbours.")
            front_weight = st.number_input("Front Neighbour", value=DEFAULT_FRONT_WEIGHT, step=1.0, format="%.1f",
                                               key='front_weight',
                                               help="Weight for repeated front neighbours.")
            diagonal_weight = st.number_input("Diagonal Neighbour", value=DEFAULT_DIAGONAL_WEIGHT, step=1.0, format="%.1f",
                                                  key='diagonal_weight',
                                                  help="Weight for repeated diagonal neighbours.")
            corner_weight = st.number_input("Corner", value=DEFAULT_CORNER_WEIGHT, step=0.1, format="%.1f",
                                                key='corner_weight',
                                                help="Exponential base for corner penalty. For example, if 5 then first corner costs 5, second 25, third 125. Thus if two corners, the cost is 5+25=30.")
            gender_weight = st.number_input("Gender", value=DEFAULT_GENDER_WEIGHT, step=0.1, format="%.1f",
                                                key='gender_weight',
                                                help="Weight for adjacent same-gender seats.")
            fixed_weight = st.number_input("Fixed Seat Diversity", value=DEFAULT_FIXED_WEIGHT, step=0.1, format="%.1f",
                                               key='fixed_weight',
                                               help="Extra weight for fixed-seat persons.")
            empty_weight = st.number_input("Empty Seat Clustering", value=DEFAULT_EMPTY_WEIGHT, step=0.1, format="%.1f",
                                               key='empty_weight',
                                               help="Weight for boundaries between empty and occupied seats.")
            preferred_side_weight = st.number_input("Preferred Side Neighbour", value=DEFAULT_PREFERRED_SIDE_WEIGHT, step=0.1, format="%.1f",
                                                        key='preferred_side_weight',
                                                        help="Penalty weight if a preferred side neighbour is missing.")
            uniformity_weight = st.number_input("Uniformity", value=DEFAULT_UNIFORMITY_WEIGHT, step=0.1, format="%.1f",
                                                key='uniformity_weight',
                                                help="Extra penalty for uneven distribution of individual costs.")
    
    st.sidebar.markdown("## Import/Export")
    uploaded_file = st.sidebar.file_uploader("Import Settings", type=['json'])
    if uploaded_file is not None:
        try:
            settings = json.load(uploaded_file)
            st.session_state.uploaded_settings = settings
            st.sidebar.success("Settings loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading settings: {str(e)}")
    
    if "show_export" not in st.session_state:
        st.session_state.show_export = False

    if st.sidebar.button("Export"):
        st.session_state.show_export = not st.session_state.show_export

    if st.session_state.show_export:
        settings = get_current_settings()
        settings_json = json.dumps(settings, indent=2)
        st.sidebar.download_button(
            label="👉 Click here to download Settings JSON",
            data=settings_json,
            file_name="seatplan_settings.json",
            mime="application/json",
            key="settings_download"
        )
        st.sidebar.write("Settings ready for download:")
        st.sidebar.json(settings)

    
    
    if run_button or "best_assignments" not in st.session_state:
        with st.spinner("Running optimization..."):
            best_assignments, best_cost, neighbors_info, corner_count, cost_history = run_optimization_and_build_data(
                iterations, initial_temp, cooling_rate,
                side_weight, front_weight, diagonal_weight,
                corner_weight,  # exponential base for corner penalty
                gender_weight, fixed_weight, empty_weight,
                people, person_genders, fixed_positions, TABLES,
                preferred_side_preferences, preferred_side_weight,
                uniformity_weight, special_cost_multipliers,
                num_rounds=num_rounds
            )
            st.session_state.best_assignments = best_assignments
            st.session_state.best_cost = best_cost
            st.session_state.neighbors_info = neighbors_info
            st.session_state.corner_count = corner_count
            st.session_state.person_genders = person_genders
            st.session_state.cost_history = cost_history
            
            combined_df = combine_all_seating_dataframes(best_assignments, TABLES, TABLE_LETTERS)
            st.session_state.combined_df = combined_df
            
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
            indiv_costs = compute_individual_cost_breakdown(
                best_assignments, compute_seat_neighbors(TABLES), TABLES,
                person_genders, fixed_positions, preferred_side_preferences,
                weights
            )
            st.session_state.indiv_costs = indiv_costs
    
    st.success(f"Optimization complete. Best cost: {st.session_state.best_cost}")

    with st.expander("Cost Breakdown"):
    
        st.header("Cost Over Iterations")
        cost_hist_df = pd.DataFrame(st.session_state.cost_history)
        cost_hist_df = cost_hist_df.set_index("iteration")
        # Now using keys that exist in our breakdown dictionary:
        st.line_chart(cost_hist_df[["total_cost", "overall_indiv_cost", "uniformity_cost" ]])
        
        st.header("Individual Cost Breakdown")
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
    

    
    with st.expander("Overall Neighbour Summary"):
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

    with st.expander("Arrangements"):
        if hasattr(st.session_state, 'combined_df'):
            st.download_button(
                label="Download Arrangements",
                data=st.session_state.combined_df.to_csv(index=False),
                file_name="seating_arrangements.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please run the optimization first to generate seating arrangements.")
        for table_id in sorted(TABLES.keys()):
            table_letter = TABLE_LETTERS[table_id]
            df = seating_dataframe_for_table(st.session_state.best_assignments, table_id, table_letter)
            st.subheader(f"Table {table_letter}")
            st.dataframe(df, height=300)
    
    if "best_assignments" in st.session_state and "neighbors_info" in st.session_state:
        NONE = "Select person..."
        # Add "None" option as the default
        person_options = [NONE] + list(st.session_state.person_genders.keys())
        selected_person = st.selectbox(
            "Select a person to highlight their seat and neighbours (optional):",
            person_options,
            index=0  # Default to "None"
        )
        
        if selected_person == NONE:
            # Display arrangements without highlighting
            for round_index, arrangement in enumerate(st.session_state.best_assignments):
                st.subheader(f"Arrangement {round_index+1}")
                for table_id in sorted(TABLES.keys()):
                    table_letter = TABLE_LETTERS[table_id]
                    html = generate_table_html_with_highlights(arrangement, table_id, table_letter, TABLES, {})
                    components.html(html, height=180, scrolling=False)
        else:
            # Use the aggregated neighbor summary for highlighting
            if selected_person in st.session_state.neighbors_info:
                aggregated_neighbors = st.session_state.neighbors_info[selected_person]
            else:
                aggregated_neighbors = {"side": set(), "front": set(), "diagonal": set()}
            display_highlighted_arrangements_by_names(selected_person, st.session_state.best_assignments,
                                                    TABLES, TABLE_LETTERS, aggregated_neighbors)
    else:
        st.info("Run the optimization first to generate seating arrangements.")

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
