import random
import string # Not explicitly used in current functions, but often useful
import uuid
import joblib
import pandas as pd
# import numpy as np # Not explicitly used in current functions, but common with pandas/sklearn
import os
import datetime
import time # For puzzle creation timestamp and cleanup
import logging
from typing import Dict, Any, Optional, List, Tuple, Set # For type hinting

# --- Initialize Logger ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if imported elsewhere
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Constants ---
MODEL_DIR = "model"
# !!! IMPORTANT: Set this to the core name of your trained model file !!!
# Example: if file is "wordlinks_ridge_regression_model.pkl", then BEST_MODEL_NAME_FROM_TRAINING = "ridge_regression"
BEST_MODEL_NAME_FROM_TRAINING = "ridge_regression" # <--- CHANGE THIS IF YOUR MODEL IS DIFFERENT
MODEL_FILENAME = os.path.join(MODEL_DIR, f"wordlinks_{BEST_MODEL_NAME_FROM_TRAINING}_model.pkl")
FEATURE_LIST_FILENAME = os.path.join(MODEL_DIR, "feature_list.pkl")

MAX_GENERATION_ATTEMPTS = 15 # Max tries to find a puzzle matching difficulty
WORDS_PER_GROUP = 4
NUM_GROUPS = 4
MAX_PUZZLE_AGE_SECONDS = 3 * 3600 # 3 hours for a non-daily puzzle to be active

# --- Load Model Pipeline and Feature Info ---
model_pipeline = None
ALL_EXPECTED_INPUT_FEATURES = [] # Features the preprocessor in the pipeline expects
TRAINING_NUMERIC_FEATURES = []
TRAINING_CATEGORICAL_FEATURES = []


try:
    if not os.path.exists(MODEL_DIR):
        logger.warning(f"Model directory '{MODEL_DIR}' not found. ML prediction will be disabled.")
    elif not os.path.exists(MODEL_FILENAME):
        logger.warning(f"Model file '{MODEL_FILENAME}' not found. ML prediction will be disabled.")
    elif not os.path.exists(FEATURE_LIST_FILENAME):
        logger.warning(f"Feature list file '{FEATURE_LIST_FILENAME}' not found. ML prediction will be disabled.")
    else:
        model_pipeline = joblib.load(MODEL_FILENAME)
        feature_info = joblib.load(FEATURE_LIST_FILENAME)
        
        TRAINING_NUMERIC_FEATURES = feature_info.get('numeric_features', [])
        TRAINING_CATEGORICAL_FEATURES = feature_info.get('categorical_features', [])
        ALL_EXPECTED_INPUT_FEATURES = TRAINING_NUMERIC_FEATURES + TRAINING_CATEGORICAL_FEATURES # Order matters!

        if not ALL_EXPECTED_INPUT_FEATURES:
            logger.error("Feature list loaded but was empty or incorrectly formatted. ML prediction disabled.")
            model_pipeline = None # Disable model if features are missing
        else:
            logger.info(f"ML Model Pipeline ({MODEL_FILENAME}) loaded successfully.")
            logger.debug(f"Expected input features for preprocessor: {ALL_EXPECTED_INPUT_FEATURES}")
            logger.debug(f"Number of expected input features: {len(ALL_EXPECTED_INPUT_FEATURES)}")

except Exception as e:
    logger.error(f"Error loading ML model pipeline or feature list: {e}. ML prediction disabled.", exc_info=True)
    model_pipeline = None
    ALL_EXPECTED_INPUT_FEATURES = []


# --- Data Definitions (From your advanced version) ---
word_categories = {
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'brown', 'gray', 'teal', 'maroon', 'indigo', 'violet'],
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'snake', 'eagle'],
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach', 'kiwi', 'watermelon', 'cherry', 'pear', 'blueberry', 'raspberry', 'lemon'],
    'vehicles': ['car', 'bus', 'train', 'airplane', 'bicycle', 'motorcycle', 'truck', 'boat', 'helicopter', 'submarine', 'scooter', 'tractor', 'van', 'ambulance', 'taxi'],
    'sports': ['soccer', 'basketball', 'tennis', 'golf', 'swimming', 'baseball', 'volleyball', 'football', 'hockey', 'rugby', 'cricket', 'boxing', 'skiing', 'climbing', 'cycling'],
    # ... (add ALL your categories back here if they were truncated in the prompt) ...
    'elements': ['fire', 'water', 'earth', 'air', 'metal', 'wood', 'electricity', 'light', 'shadow', 'ice', 'steam', 'plasma', 'space', 'void', 'energy']
}

connection_types = { # Complexity values
    'same_category': 1, 'begins_with': 2, 'ends_with': 3, 'syllable_count': 4,
    'synonym_groups': 5, 'antonym_groups': 6, 'compound_words': 7, 'rhyming_words': 7,
    'conceptual_relation': 8, 'multiple_rules': 10, 'letter_pattern': 5, 'anagrams': 9,
    'homophones': 6, 'contains_substring': 4, 'metaphorical_relation': 9
}

connection_descriptions = {
    'same_category': "Words in the Same Category", 'begins_with': "Words Starting With Same Letter",
    'ends_with': "Words Ending With Same Letter", 'syllable_count': "Same Number of Syllables",
    'synonym_groups': "Synonyms", 'antonym_groups': "Antonyms",
    'compound_words': "Form Compound Words", 'rhyming_words': "Rhyming Words",
    'conceptual_relation': "Conceptual Relationship", 'multiple_rules': "Connected by Multiple Rules",
    'letter_pattern': "Shared Letter Pattern", 'anagrams': "Anagrams",
    'homophones': "Homophones", 'contains_substring': "Contain Same Substring",
    'metaphorical_relation': "Metaphorical Relation"
}

word_rarity_levels = { # Rarity values
    'very_common': 1, 'common': 2, 'somewhat_common': 4, 'uncommon': 6,
    'rare': 8, 'very_rare': 10, 'extremely_rare': 12
}

# --- In-memory storage for active puzzles ---
active_puzzles: Dict[str, Dict[str, Any]] = {}


# --- ML Prediction Helper ---
def get_fallback_prediction(params: Dict[str, Any]) -> Dict[str, Any]:
    """Provides a default prediction structure when ML fails or is disabled."""
    logger.warning("[FALLBACK] Using rule-based difficulty estimation.")
    complexity = connection_types.get(params.get('connection_type', 'same_category'), 1)
    rarity = word_rarity_levels.get(params.get('word_rarity', 'common'), 2)
    num_words_in_puzzle = params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP)

    # Simplified heuristic for fallback difficulty
    # (Adjust these thresholds based on your game's feel)
    score = complexity + rarity
    if num_words_in_puzzle > 16: score += (num_words_in_puzzle - 16) # Penalty for too many words (if applicable)
    
    # This is a very rough estimate for solve time based on score
    est_time = 20 + score * 5 

    difficulty = "medium" # Default
    if score <= 5: difficulty = "easy" # Example: (cat_1 + common_2 = 3)
    elif score <= 12: difficulty = "medium" # Example: (conceptual_8 + uncommon_6 = 14, but might be medium)
    else: difficulty = "hard"
    
    logger.info(f"[FALLBACK] Params: {params}, Calculated Score: {score}, Est. Time: {est_time}, Fallback Difficulty: {difficulty}")
    return {'predicted_solve_time': round(est_time, 2), 'difficulty': difficulty, 'is_fallback': True}


def predict_difficulty_for_params(params: Dict[str, Any]) -> Dict[str, Any]:
    if not model_pipeline or not ALL_EXPECTED_INPUT_FEATURES:
        logger.warning("ML model or feature list not available. Using fallback prediction.")
        return get_fallback_prediction(params)

    connection_complexity = connection_types.get(params['connection_type'], 5) # Default if type is exotic
    word_rarity_value = word_rarity_levels.get(params['word_rarity'], 5) # Default if rarity is exotic
    current_hour = datetime.datetime.now().hour
    num_words_val = params.get('num_words', float(NUM_GROUPS * WORDS_PER_GROUP))

    # --- Create DataFrame matching the TRAINING 'X' structure ---
    # Initialize with default values for all expected features
    # This ensures all columns are present even if not directly set by `params`
    feature_data_dict = {feature_name: 0.0 for feature_name in ALL_EXPECTED_INPUT_FEATURES}

    # Update with values from `params` or derived values
    feature_data_dict.update({
        'num_words': float(num_words_val),
        'connection_complexity': float(connection_complexity),
        'word_rarity_value': float(word_rarity_value),
        'semantic_distance': float(params.get('semantic_distance', 5.0)), # Default if not in params
        'time_of_day': float(current_hour),
        'hints_used': float(params.get('hints_used', 0)), # Usually 0 for generation

        # Default values for features not directly controlled by generation params
        # These should ideally match the typical or average values seen during training
        # or be set to neutral values if their impact during generation is unknown.
        'num_players': 50.0, # Example default
        'completions': 40.0, # Example default
        'completion_rate': 80.0, # Example default
        'attempt_count': 2.0, # Example default (average attempts for a group)
        'time_before_first_attempt': 10.0, # Example default
        'hover_count': float(num_words_val * 1.5), # Example default (words * typical hovers)
        'abandonment_rate': 20.0, # Example default
        'competitiveness_score': 5.0, # Example default
        'frustration_score': 3.0, # Example default
        'satisfaction_score': 7.0, # Example default
        'learning_value': 5.0, # Example default
        'engagement_score': 6.0, # Example default
        'replayability_score': 4.0, # Example default
        'avg_attempts_before_success': 1.5 # Example default
    })
    
    # Filter down to only expected features to avoid extra columns issues
    final_feature_data = {key: feature_data_dict[key] for key in ALL_EXPECTED_INPUT_FEATURES if key in feature_data_dict}
    
    # Check if any expected features are missing after filtering (should not happen if defaults are set for all)
    missing_features = set(ALL_EXPECTED_INPUT_FEATURES) - set(final_feature_data.keys())
    if missing_features:
        logger.error(f"Critical error: Missing expected features for prediction after preparing data: {missing_features}")
        return get_fallback_prediction(params)

    try:
        # Create DataFrame with columns in the exact order expected by the pipeline
        predict_df = pd.DataFrame([final_feature_data], columns=ALL_EXPECTED_INPUT_FEATURES)
        logger.debug(f"DataFrame sent to pipeline:\n{predict_df.to_string()}")
    except Exception as e:
        logger.error(f"Error creating DataFrame for prediction: {e}", exc_info=True)
        return get_fallback_prediction(params)

    try:
        predicted_time_array = model_pipeline.predict(predict_df)
        predicted_time = float(predicted_time_array[0]) # Ensure it's a float
    except ValueError as e:
         logger.error(f"ValueError during pipeline prediction: {e}. Data:\n{predict_df.iloc[0].to_dict()}", exc_info=True)
         return get_fallback_prediction(params)
    except Exception as e:
        logger.error(f"Unexpected error during pipeline prediction: {e}. Data:\n{predict_df.iloc[0].to_dict()}", exc_info=True)
        return get_fallback_prediction(params)

    difficulty = "medium" # Default
    if predicted_time < 40: difficulty = "easy"
    elif predicted_time < 80: difficulty = "medium"
    else: difficulty = "hard"
    
    logger.info(f"ML Prediction: Solve Time={predicted_time:.2f}s, Difficulty='{difficulty}' for params: {params}")
    return {'predicted_solve_time': round(predicted_time, 2), 'difficulty': difficulty, 'is_fallback': False}


# --- Puzzle Generation Logic ---
def _get_all_available_words() -> Set[str]:
    return set(word.lower() for cat_words in word_categories.values() for word in cat_words)

def _generate_fallback_groups(num_words_needed: int, connection_type_key: str) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]:
    """Generates random groups from categories as a fallback. Returns solution, all_words, descriptions, difficulty_indices."""
    solution_groups: Dict[str, List[str]] = {}
    all_words_list: List[str] = []
    group_descriptions: Dict[str, str] = {}
    difficulty_indices: Dict[str, int] = {} # For coloring

    # Use 'same_category' as the most reliable fallback type
    logger.warning(f"[FALLBACK-GROUPS] Using 'same_category' for word selection due to issues with '{connection_type_key}'.")
    
    if len(word_categories) < NUM_GROUPS:
        raise ValueError(f"Not enough word categories ({len(word_categories)}) defined for fallback generation.")

    selected_category_names = random.sample(list(word_categories.keys()), NUM_GROUPS)
    
    used_words_overall = set()

    for i, category_name in enumerate(selected_category_names):
        available_words_in_cat = [w.lower() for w in word_categories.get(category_name, []) if w.lower() not in used_words_overall]
        
        if len(available_words_in_cat) < WORDS_PER_GROUP:
            logger.warning(f"[FALLBACK-GROUPS] Category '{category_name}' has < {WORDS_PER_GROUP} unused words. Trying to supplement or skipping.")
            # Attempt to supplement with any globally available unused words (less ideal but better than failing)
            global_available_unused = [w for w in list(_get_all_available_words() - used_words_overall) if w not in available_words_in_cat]
            needed_supplement = WORDS_PER_GROUP - len(available_words_in_cat)
            if len(global_available_unused) >= needed_supplement:
                available_words_in_cat.extend(random.sample(global_available_unused, needed_supplement))
            else: # Still not enough, this group will be short or skipped
                 logger.error(f"[FALLBACK-GROUPS] Cannot form full group for '{category_name}'. Available: {len(available_words_in_cat)}")
                 if not available_words_in_cat : continue # Skip if empty
            
        group_words = random.sample(available_words_in_cat, min(WORDS_PER_GROUP, len(available_words_in_cat)))
        
        if len(group_words) == WORDS_PER_GROUP: # Only add if a full group was formed
            group_id = f"group_{i+1}"
            solution_groups[group_id] = sorted(group_words) # Already lowercase
            all_words_list.extend(group_words)
            group_descriptions[group_id] = connection_descriptions.get('same_category', "Related Words") + f": {category_name.capitalize()}"
            difficulty_indices[group_id] = i # Simple index for fallback coloring
            used_words_overall.update(group_words)
        else:
            logger.warning(f"Could not form a complete group of {WORDS_PER_GROUP} words for category {category_name} in fallback.")


    if len(solution_groups) < NUM_GROUPS:
        raise ValueError(f"Fallback group generation failed to produce {NUM_GROUPS} complete groups. Produced: {len(solution_groups)}")

    return solution_groups, all_words_list, group_descriptions, difficulty_indices


def generate_solvable_puzzle(target_difficulty: str) -> Dict[str, Any]:
    puzzle_id = str(uuid.uuid4())
    num_words_total = NUM_GROUPS * WORDS_PER_GROUP
    generation_params: Dict[str, Any] = {}
    predicted_result: Dict[str, Any] = {}
    
    logger.info(f"\n--- Attempting to generate puzzle for difficulty: '{target_difficulty}' ---")

    # Attempt to find parameters that match the target difficulty using the ML model
    found_matching_params = False
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        # Define parameter ranges based on target difficulty (as in your advanced logic)
        if target_difficulty == 'easy':
            conn_type_name = random.choice(['same_category', 'begins_with', 'ends_with', 'contains_substring'])
            word_rarity_name = random.choice(['very_common', 'common'])
            semantic_distance = random.uniform(1, 4)
        elif target_difficulty == 'medium':
            conn_type_name = random.choice(['syllable_count', 'letter_pattern', 'rhyming_words', 'synonym_groups', 'homophones', 'antonym_groups', 'compound_words'])
            word_rarity_name = random.choice(['common', 'somewhat_common', 'uncommon'])
            semantic_distance = random.uniform(3, 7)
        else:  # hard
            conn_type_name = random.choice(['conceptual_relation', 'anagrams', 'metaphorical_relation', 'multiple_rules'])
            word_rarity_name = random.choice(['uncommon', 'rare', 'very_rare']) # Removed 'extremely_rare' if too hard to generate words for
            semantic_distance = random.uniform(6, 10)

        if conn_type_name not in connection_types:
            logger.warning(f"Attempt {attempt+1}: Conn type '{conn_type_name}' not in connection_types map. Skipping.")
            continue
        if word_rarity_name not in word_rarity_levels:
            logger.warning(f"Attempt {attempt+1}: Rarity '{word_rarity_name}' not in word_rarity_levels map. Skipping.")
            continue
            
        current_generation_params = {
            'num_words': num_words_total,
            'connection_type': conn_type_name,
            'word_rarity': word_rarity_name,
            'semantic_distance': semantic_distance,
        }
        
        current_predicted_result = predict_difficulty_for_params(current_generation_params)
        logger.info(f"Attempt {attempt+1}: Params: {conn_type_name}/{word_rarity_name} (SD:{semantic_distance:.1f}), Predicted: {current_predicted_result.get('difficulty')} ({current_predicted_result.get('predicted_solve_time')}s)")

        if current_predicted_result.get('difficulty') == target_difficulty:
            generation_params = current_generation_params
            predicted_result = current_predicted_result
            found_matching_params = True
            logger.info(f"--> Found suitable parameters for '{target_difficulty}' on attempt {attempt+1}.")
            break
        elif attempt == 0 or not generation_params: # Store first valid prediction as a fallback
            generation_params = current_generation_params
            predicted_result = current_predicted_result

    if not found_matching_params:
        logger.warning(f"Could not find exact match for '{target_difficulty}' after {MAX_GENERATION_ATTEMPTS} attempts. Using best found or fallback.")
        if not generation_params: # If no valid prediction was ever made
            logger.error("No valid prediction could be made. Defaulting params for fallback generation.")
            generation_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total, 'semantic_distance': 3.0}
            predicted_result = get_fallback_prediction(generation_params)


    # --- Word Selection / Puzzle Assembly ---
    final_connection_type = generation_params.get('connection_type', 'same_category')
    solution_groups: Dict[str, List[str]] = {}
    all_words_for_grid: List[str] = []
    group_descriptions: Dict[str, str] = {}
    difficulty_index_map_for_puzzle: Dict[str, int] = {} # For coloring
    generation_successful = False
    
    logger.info(f"--- Finalizing puzzle words with connection type: '{final_connection_type}' ---")

    try:
        # TODO: Implement robust word selection for EACH connection_type.
        # The following are simplified examples or placeholders.
        if final_connection_type == 'same_category':
            if len(word_categories) < NUM_GROUPS: raise ValueError("Not enough categories for 'same_category'")
            selected_cat_names = random.sample(list(word_categories.keys()), NUM_GROUPS)
            used_words_overall = set()
            for i, cat_name in enumerate(selected_cat_names):
                group_id = f"group_{i+1}"
                possible_words = [w.lower() for w in word_categories[cat_name] if w.lower() not in used_words_overall]
                if len(possible_words) < WORDS_PER_GROUP:
                    logger.warning(f"Category '{cat_name}' has insufficient unique words. Need {WORDS_PER_GROUP}, have {len(possible_words)}.")
                    continue # Skip this category or attempt fallback for this group
                
                group_words = random.sample(possible_words, WORDS_PER_GROUP)
                solution_groups[group_id] = sorted(group_words)
                all_words_for_grid.extend(group_words)
                used_words_overall.update(group_words)
                group_descriptions[group_id] = connection_descriptions[final_connection_type] + f": {cat_name.capitalize()}"
                difficulty_index_map_for_puzzle[group_id] = i # Simple ordering for now
            if len(solution_groups) == NUM_GROUPS: generation_successful = True

        # Add more `elif final_connection_type == '...':` blocks for other types.
        # Each block should populate solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle
        # and set generation_successful = True if it forms NUM_GROUPS complete groups.

        if not generation_successful:
            logger.warning(f"Specific logic for '{final_connection_type}' failed or not implemented. Using fallback group generation.")
            solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
                _generate_fallback_groups(num_words_total, final_connection_type)
            # Fallback ensures generation_successful if it doesn't raise an error

    except Exception as e:
        logger.error(f"Exception during word selection for '{final_connection_type}': {e}. Attempting full fallback.", exc_info=True)
        try:
            solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
                _generate_fallback_groups(num_words_total, "same_category") # Safest fallback
        except Exception as fallback_e:
            logger.critical(f"Full fallback generation also failed: {fallback_e}", exc_info=True)
            raise ValueError("Failed to generate puzzle words after multiple fallbacks.") from fallback_e

    if len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
        logger.error(f"Final puzzle assembly error. Words: {len(all_words_for_grid)}/{num_words_total}, Groups: {len(solution_groups)}/{NUM_GROUPS}")
        raise ValueError("Generated puzzle has incorrect number of words or groups.")

    random.shuffle(all_words_for_grid)

    active_puzzles[puzzle_id] = {
        "puzzle_id": puzzle_id,
        "words_on_grid": [word.upper() for word in all_words_for_grid], # Frontend expects uppercase
        "solution": {key: [word.upper() for word in words] for key, words in solution_groups.items()},
        "descriptions": group_descriptions,
        "difficulty": predicted_result.get('difficulty', target_difficulty), # Use predicted or target
        "predicted_solve_time": predicted_result.get('predicted_solve_time', -1),
        "parameters": {**generation_params, "difficulty_index_map": difficulty_index_map_for_puzzle},
        "creation_time": datetime.datetime.now()
    }
    logger.info(f"--- Puzzle {puzzle_id} (Target: {target_difficulty}, Predicted: {predicted_result.get('difficulty')}) generated. ---")

    return {
        "puzzle_id": puzzle_id,
        "words": active_puzzles[puzzle_id]["words_on_grid"],
        "difficulty": active_puzzles[puzzle_id]["difficulty"] # Send the determined difficulty
    }


# --- Daily Challenge Logic (Integrated) ---
def get_or_generate_daily_challenge() -> dict:
    cleanup_old_puzzles()
    today_str = datetime.date.today().isoformat()
    daily_puzzle_id = f"daily_{today_str}"

    if daily_puzzle_id not in active_puzzles:
        logger.info(f"Generating new daily challenge for {today_str} (ID: {daily_puzzle_id})")
        original_random_state = random.getstate()
        random.seed(today_str) # Seed for deterministic daily puzzle

        # Generate a puzzle (e.g., "medium" difficulty for daily)
        # This calls your main generate_solvable_puzzle which uses ML (or its fallback)
        temp_puzzle_client_data = generate_solvable_puzzle(difficulty="medium")
        temp_puzzle_id_from_gen = temp_puzzle_client_data["puzzle_id"]

        if temp_puzzle_id_from_gen in active_puzzles:
            daily_puzzle_server_data = active_puzzles.pop(temp_puzzle_id_from_gen)
            
            daily_puzzle_server_data["puzzle_id"] = daily_puzzle_id # Override with daily ID
            daily_puzzle_server_data["is_daily"] = True
            daily_puzzle_server_data["difficulty"] = "Daily Challenge" # Override display difficulty
            
            active_puzzles[daily_puzzle_id] = daily_puzzle_server_data
            logger.info(f"Daily challenge {daily_puzzle_id} (based on temp {temp_puzzle_id_from_gen}) stored.")
        else:
            logger.error(f"Failed to find temp puzzle {temp_puzzle_id_from_gen} in active_puzzles for daily.")
            random.setstate(original_random_state) # Restore before erroring
            raise ValueError("Daily challenge temp puzzle not found in active_puzzles after generation.")
        
        random.setstate(original_random_state) # Restore random state
    else:
        logger.info(f"Returning existing daily challenge for {today_str} (ID: {daily_puzzle_id})")

    # Prepare client response
    puzzle_to_send = active_puzzles[daily_puzzle_id]
    return {
        "puzzle_id": puzzle_to_send["puzzle_id"],
        "words": puzzle_to_send["words_on_grid"],
        "difficulty": puzzle_to_send["difficulty"], # This will be "Daily Challenge"
        "is_daily": True
    }


# --- Answer Checking Logic (From your advanced version) ---
def check_puzzle_answer(puzzle_id: str, user_groups_attempt: Dict[str, List[str]]) -> Dict[str, Any]:
    if puzzle_id not in active_puzzles:
        logger.warning(f"Answer check for invalid/expired puzzle ID: {puzzle_id}")
        return {"correct": False, "message": "Invalid or expired puzzle.", "solved_groups": {}}

    puzzle_data = active_puzzles[puzzle_id]
    actual_solution = puzzle_data["solution"] # Expected: {"group_1": ["W1", "W2", "W3", "W4"], ...} (uppercase, sorted)
    actual_descriptions = puzzle_data["descriptions"]
    difficulty_index_map = puzzle_data.get("parameters", {}).get("difficulty_index_map", {})

    attempted_words_key = next(iter(user_groups_attempt), None)
    if not attempted_words_key:
        return {"correct": False, "message": "No attempt data provided.", "solved_groups": {}}
    
    # Normalize user's attempt: uppercase and sort
    attempted_words_sorted_upper = sorted([word.upper() for word in user_groups_attempt[attempted_words_key]])

    if len(attempted_words_sorted_upper) != WORDS_PER_GROUP:
        return {"correct": False, "message": f"Please select exactly {WORDS_PER_GROUP} words.", "solved_groups": {}}

    for group_key, correct_words_sorted_upper in actual_solution.items():
        if attempted_words_sorted_upper == correct_words_sorted_upper:
            logger.info(f"Correct group '{group_key}' found for puzzle {puzzle_id}")
            return {
                "correct": True,
                "message": f"Correct! Category: {actual_descriptions.get(group_key, 'Group Found')}",
                "solved_groups": {
                    group_key: {
                        "description": actual_descriptions.get(group_key, "Unknown Category"),
                        "difficulty_index": difficulty_index_map.get(group_key, 0) 
                    }
                }
            }
    
    logger.info(f"Incorrect attempt for puzzle {puzzle_id}: {attempted_words_sorted_upper}")
    return {"correct": False, "message": "That's not a correct group. Try again!", "solved_groups": {}}


# --- Hint Logic (From your advanced version, ensure it returns "words") ---
def get_puzzle_hint(puzzle_id: str, solved_group_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    if solved_group_keys is None: solved_group_keys = []
    if puzzle_id not in active_puzzles:
        logger.warning(f"Hint for invalid/expired puzzle ID: {puzzle_id}")
        return {"hint": None, "message": "Invalid or expired puzzle.", "words": []}

    puzzle_data = active_puzzles[puzzle_id]
    actual_solution = puzzle_data.get("solution", {})
    actual_descriptions = puzzle_data.get("descriptions", {})

    unsolved_groups = {}
    for group_key, words_in_group in actual_solution.items():
        if group_key not in solved_group_keys:
            unsolved_groups[group_key] = {
                "words": words_in_group, # These are already uppercase from generation
                "description": actual_descriptions.get(group_key, "A Group")
            }
    
    if not unsolved_groups:
        return {"hint": None, "message": "All groups already solved!", "words": []}

    target_group_key = random.choice(list(unsolved_groups.keys()))
    target_group_info = unsolved_groups[target_group_key]
    
    words_for_hint_highlight = []
    if len(target_group_info["words"]) >= 2:
        words_for_hint_highlight = random.sample(target_group_info["words"], 2)
    elif target_group_info["words"]:
        words_for_hint_highlight = random.sample(target_group_info["words"], 1)
    
    hint_text_core = f"Consider the category: '{target_group_info['description']}'."
    if words_for_hint_highlight:
         hint_text_core += f" Two words in this group are '{words_for_hint_highlight[0]}' and '{words_for_hint_highlight[1]}'." if len(words_for_hint_highlight) >=2 else f" One word is '{words_for_hint_highlight[0]}'."


    logger.info(f"Providing hint for puzzle {puzzle_id}: Category '{target_group_info['description']}'")
    return {
        "hint": hint_text_core,
        "words": words_for_hint_highlight, # For frontend highlighting
        "message": "Hint provided."
    }

# --- Cleanup Function (From your advanced version, adapted) ---
def cleanup_old_puzzles():
    current_time_dt = datetime.datetime.now()
    today_date = datetime.date.today()
    puzzles_to_delete = []

    for pid, data in list(active_puzzles.items()): # Iterate over a copy for safe deletion
        creation_time_dt = data.get("creation_time")
        if not isinstance(creation_time_dt, datetime.datetime): # Handle old format if any
            if isinstance(creation_time_dt, float):
                creation_time_dt = datetime.datetime.fromtimestamp(creation_time_dt)
            else: # Cannot determine age, skip or log
                logger.warning(f"Puzzle {pid} has invalid creation_time format: {creation_time_dt}. Skipping cleanup for this one.")
                continue
        
        if pid.startswith("daily_"):
            puzzle_date_str = pid.replace("daily_", "")
            try:
                puzzle_date = datetime.date.fromisoformat(puzzle_date_str)
                if puzzle_date < today_date:
                    puzzles_to_delete.append(pid)
            except ValueError: # Should not happen if daily_id format is consistent
                logger.error(f"Malformed daily puzzle ID for cleanup: {pid}")
        elif (current_time_dt - creation_time_dt).total_seconds() > MAX_PUZZLE_AGE_SECONDS:
            puzzles_to_delete.append(pid)

    for pid in puzzles_to_delete:
        if pid in active_puzzles:
            del active_puzzles[pid]
            logger.info(f"Cleaned up old/expired puzzle: {pid}")

# --- Example Usage (for testing this file directly) ---
if __name__ == "__main__":
    logger.info("Testing puzzle_logic.py...")
    # Test 1: Regular puzzle generation
    try:
        print("\n--- Testing Regular Puzzle Generation (Easy) ---")
        easy_puzzle = generate_solvable_puzzle(target_difficulty="easy")
        print(f"Easy Puzzle Client Data: {easy_puzzle}")
        if easy_puzzle['puzzle_id'] in active_puzzles:
            print(f"Easy Puzzle Server Data (sample): solution keys {list(active_puzzles[easy_puzzle['puzzle_id']]['solution'].keys())}")
    except Exception as e:
        print(f"Error generating easy puzzle: {e}")

    # Test 2: Daily Challenge
    try:
        print("\n--- Testing Daily Challenge Generation ---")
        daily_data_client = get_or_generate_daily_challenge()
        print(f"Daily Challenge Client Data: {daily_data_client}")
        daily_id = daily_data_client['puzzle_id']
        if daily_id in active_puzzles:
            print(f"Daily Challenge Server Data: ID={active_puzzles[daily_id]['puzzle_id']}, IsDaily={active_puzzles[daily_id]['is_daily']}")
        # Call again to test retrieval
        daily_data_client_2 = get_or_generate_daily_challenge()
        assert daily_data_client_2["puzzle_id"] == daily_id, "Daily challenge ID should be consistent for the day"
        print("Second call for daily challenge returned same ID, as expected.")
    except Exception as e:
        print(f"Error with daily challenge: {e}")

    # Test 3: Answer Checking (using the daily puzzle if available)
    if 'daily_id' in locals() and daily_id in active_puzzles:
        print("\n--- Testing Answer Checking (with Daily Puzzle) ---")
        daily_solution = active_puzzles[daily_id]["solution"]
        first_group_key = next(iter(daily_solution))
        correct_words = daily_solution[first_group_key]
        
        correct_attempt = {"attempt": correct_words}
        result_correct = check_puzzle_answer(daily_id, correct_attempt)
        print(f"Correct attempt result: {result_correct}")
        assert result_correct["correct"] is True

        incorrect_attempt = {"attempt": ["WRONG", "WORDS", "HERE", "NOW"]}
        result_incorrect = check_puzzle_answer(daily_id, incorrect_attempt)
        print(f"Incorrect attempt result: {result_incorrect}")
        assert result_incorrect["correct"] is False
    
    # Test 4: Hint (using daily puzzle)
    if 'daily_id' in locals() and daily_id in active_puzzles:
         print("\n--- Testing Hint (with Daily Puzzle) ---")
         hint = get_puzzle_hint(daily_id, [])
         print(f"Hint: {hint}")
         assert hint.get("hint") is not None
         assert isinstance(hint.get("words"), list)

    # Test 5: Cleanup
    print("\n--- Testing Cleanup (simulating old puzzle) ---")
    test_cleanup_id = "test_cleanup_puzzle"
    active_puzzles[test_cleanup_id] = {
        "creation_time": datetime.datetime.now() - datetime.timedelta(seconds=MAX_PUZZLE_AGE_SECONDS + 60),
        "solution": {}, "descriptions": {}, "parameters": {} # Minimal data for cleanup
    }
    print(f"Before cleanup, {test_cleanup_id} active: {test_cleanup_id in active_puzzles}")
    cleanup_old_puzzles()
    print(f"After cleanup, {test_cleanup_id} active: {test_cleanup_id in active_puzzles}")
    assert test_cleanup_id not in active_puzzles

    logger.info("puzzle_logic.py tests completed.")