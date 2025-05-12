import random
import string
import uuid
import joblib
import pandas as pd
import os
import datetime
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Set

# --- Initialize Logger ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

# --- Constants ---
MODEL_DIR = "model"
BEST_MODEL_NAME_FROM_TRAINING = "ridge_regression"  # <--- !!! UPDATE THIS TO MATCH YOUR MODEL !!!
MODEL_FILENAME = os.path.join(MODEL_DIR, f"wordlinks_{BEST_MODEL_NAME_FROM_TRAINING}_model.pkl")
FEATURE_LIST_FILENAME = os.path.join(MODEL_DIR, "feature_list.pkl")

MAX_GENERATION_ATTEMPTS = 25
WORDS_PER_GROUP = 4
NUM_GROUPS = 4
MAX_PUZZLE_AGE_SECONDS = 3 * 3600

# --- Load Model Pipeline and Feature Info ---
model_pipeline = None
ALL_EXPECTED_INPUT_FEATURES = []
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
        ALL_EXPECTED_INPUT_FEATURES = TRAINING_NUMERIC_FEATURES + TRAINING_CATEGORICAL_FEATURES
        if not ALL_EXPECTED_INPUT_FEATURES:
            logger.error("Feature list empty. ML disabled.")
            model_pipeline = None
        else:
            logger.info(f"ML Model Pipeline ({MODEL_FILENAME}) loaded successfully.")
            logger.debug(f"Expected input features (order matters): {ALL_EXPECTED_INPUT_FEATURES}")
except Exception as e:
    logger.error(f"Error loading ML model/features: {e}. ML disabled.", exc_info=True)
    model_pipeline = None
    ALL_EXPECTED_INPUT_FEATURES = []

# --- Data Definitions ---
word_categories = {
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'brown', 'gray', 'teal', 'maroon', 'indigo', 'violet', 'cyan', 'magenta', 'lime', 'olive'],
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'snake', 'eagle', 'shark', 'whale', 'dolphin', 'penguin'],
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach', 'kiwi', 'watermelon', 'cherry', 'pear', 'blueberry', 'raspberry', 'lemon', 'lime', 'plum', 'apricot'],
    'vehicles': ['car', 'bus', 'train', 'airplane', 'bicycle', 'motorcycle', 'truck', 'boat', 'helicopter', 'submarine', 'scooter', 'tractor', 'van', 'ambulance', 'taxi', 'rocket', 'ferry', 'yacht'],
    'sports': ['soccer', 'basketball', 'tennis', 'golf', 'swimming', 'baseball', 'volleyball', 'football', 'hockey', 'rugby', 'cricket', 'boxing', 'skiing', 'climbing', 'cycling', 'surfing', 'karate', 'judo'],
    'countries': ['usa', 'canada', 'japan', 'brazil', 'france', 'germany', 'india', 'australia', 'china', 'mexico', 'italy', 'spain', 'russia', 'egypt', 'kenya', 'argentina', 'nigeria', 'greece'],
    'professions': ['doctor', 'teacher', 'engineer', 'lawyer', 'chef', 'artist', 'scientist', 'writer', 'nurse', 'programmer', 'architect', 'accountant', 'journalist', 'pilot', 'firefighter', 'musician', 'actor', 'police'],
    'instruments': ['piano', 'guitar', 'violin', 'drums', 'flute', 'trumpet', 'saxophone', 'clarinet', 'harp', 'cello', 'banjo', 'harmonica', 'trombone', 'accordion', 'ukulele', 'xylophone', 'bass', 'keyboard'],
    'weather': ['sunny', 'rainy', 'cloudy', 'snowy', 'windy', 'stormy', 'foggy', 'humid', 'dry', 'freezing', 'tropical', 'mild', 'chilly', 'breezy', 'thunderous', 'drizzle', 'hail', 'blizzard'],
    'emotions': ['happy', 'sad', 'angry', 'excited', 'fearful', 'surprised', 'disgusted', 'anxious', 'calm', 'bored', 'content', 'proud', 'envious', 'jealous', 'grateful', 'hopeful', 'lonely', 'loved'],
    'furniture': ['table', 'chair', 'couch', 'dresser', 'bed', 'desk', 'bookcase', 'stool', 'cabinet', 'wardrobe', 'bench', 'ottoman', 'armchair', 'nightstand', 'futon', 'shelf', 'lamp', 'mirror'],
    'body_parts': ['head', 'arm', 'leg', 'foot', 'hand', 'eye', 'ear', 'nose', 'mouth', 'finger', 'elbow', 'knee', 'shoulder', 'neck', 'ankle', 'wrist', 'toe', 'chest'],
    'foods': ['pizza', 'pasta', 'burger', 'salad', 'soup', 'sandwich', 'taco', 'rice', 'bread', 'cheese', 'fish', 'steak', 'cake', 'cookie', 'chocolate', 'sushi', 'curry', 'fries'],
    'clothing': ['shirt', 'pants', 'dress', 'skirt', 'jacket', 'coat', 'hat', 'shoe', 'sock', 'scarf', 'glove', 'belt', 'tie', 'sweater', 'shorts', 'boots', 'sandals', 'jeans'],
    'elements': ['fire', 'water', 'earth', 'air', 'metal', 'wood', 'electricity', 'light', 'shadow', 'ice', 'steam', 'plasma', 'space', 'void', 'energy', 'sound', 'time', 'gravity'],
    'shapes': ['circle', 'square', 'triangle', 'rectangle', 'oval', 'star', 'heart', 'diamond', 'cube', 'sphere', 'pyramid', 'cylinder', 'pentagon', 'hexagon', 'octagon'],
    'kitchen_items': ['knife', 'fork', 'spoon', 'plate', 'bowl', 'cup', 'pan', 'pot', 'oven', 'microwave', 'blender', 'toaster', 'kettle', 'grater', 'whisk']
}
connection_types = {
    'same_category': 1, 'begins_with': 2, 'ends_with': 3, 'syllable_count': 4,
    'synonym_groups': 5, 'antonym_groups': 6, 'compound_words': 7, 'rhyming_words': 7,
    'conceptual_relation': 8, 'multiple_rules': 10, 'letter_pattern': 5, 'anagrams': 9,
    'homophones': 6, 'contains_substring': 4, 'metaphorical_relation': 9
}
connection_descriptions = {
    'same_category': "Same Category", 'begins_with': "Begin With Same Letter",
    'ends_with': "End With Same Letter", 'syllable_count': "Same Syllable Count",
    'synonym_groups': "Synonyms", 'antonym_groups': "Antonyms",
    'compound_words': "Compound Words", 'rhyming_words': "Rhyming Words",
    'conceptual_relation': "Conceptual Link", 'multiple_rules': "Multiple Connections",
    'letter_pattern': "Shared Letters", 'anagrams': "Anagrams",
    'homophones': "Homophones", 'contains_substring': "Contain Substring",
    'metaphorical_relation': "Metaphorical Link"
}
word_rarity_levels = {
    'very_common': 1, 'common': 2, 'somewhat_common': 4, 'uncommon': 6,
    'rare': 8, 'very_rare': 10, 'extremely_rare': 12
}
active_puzzles: Dict[str, Dict[str, Any]] = {}

# --- Helper Functions (MOVED UP and DEFINED) ---
def _get_all_available_words(exclude_words: Optional[Set[str]] = None) -> List[str]:
    """Helper to get a list of all unique words from word_categories, optionally excluding some."""
    if exclude_words is None: exclude_words = set()
    all_words = set()
    for category_words in word_categories.values():
        for word in category_words:
            all_words.add(word.lower())
    return list(all_words - exclude_words)

def _generate_groups_by_category(num_groups_to_generate: int, words_per_group_val: int,
                                 exclude_words_from_puzzle: Set[str]) -> Optional[Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]]:
    """
    Attempts to generate a specified number of groups based on 'same_category'.
    Tries to ensure words within a group are unique to that group for this puzzle instance.
    Returns: (solution_groups, all_words_list, group_descriptions, difficulty_indices) or None if failed.
    """
    solution_groups: Dict[str, List[str]] = {}
    all_words_list: List[str] = []
    group_descriptions: Dict[str, str] = {}
    difficulty_indices: Dict[str, int] = {}

    eligible_categories = {}
    for cat_name, cat_words_list in word_categories.items():
        potential_words = [w.lower() for w in cat_words_list if w.lower() not in exclude_words_from_puzzle]
        if len(set(potential_words)) >= words_per_group_val:
            eligible_categories[cat_name] = list(set(potential_words))

    if len(eligible_categories) < num_groups_to_generate:
        logger.warning(f"Not enough eligible categories ({len(eligible_categories)}) with sufficient unique words to form {num_groups_to_generate} groups.")
        return None

    selected_category_names = random.sample(list(eligible_categories.keys()), num_groups_to_generate)
    words_used_in_this_generation_call = set() 

    for i, category_name in enumerate(selected_category_names):
        group_id = f"group_{i+1}"
        words_for_this_group_selection = [w for w in eligible_categories[category_name] if w not in words_used_in_this_generation_call]

        if len(words_for_this_group_selection) < words_per_group_val:
            logger.warning(f"Category '{category_name}' ran out of unique words for group {group_id}. Needed {words_per_group_val}, have {len(words_for_this_group_selection)}.")
            return None 
            
        selected_words_for_group = random.sample(words_for_this_group_selection, words_per_group_val)
        
        solution_groups[group_id] = sorted(selected_words_for_group)
        all_words_list.extend(selected_words_for_group)
        group_descriptions[group_id] = connection_descriptions.get('same_category', "Category") + f": {category_name.capitalize()}"
        difficulty_indices[group_id] = i
        words_used_in_this_generation_call.update(selected_words_for_group)

    if len(solution_groups) == num_groups_to_generate:
        return solution_groups, all_words_list, group_descriptions, difficulty_indices
    else:
        logger.error(f"Failed to generate {num_groups_to_generate} category-based groups. Generated {len(solution_groups)}.")
        return None

def _generate_fallback_groups(num_words_needed: int, connection_type_key_attempted: str) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]:
    """Generates random groups primarily based on 'same_category' as a fallback."""
    # This now correctly calls _generate_groups_by_category for its core logic
    logger.warning(f"[FALLBACK-GROUPS] Using specific 'same_category' logic for '{connection_type_key_attempted}'.")
    category_based_fallback_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set())
    if category_based_fallback_data:
        return category_based_fallback_data
    else:
        # This would be a more severe fallback if _generate_groups_by_category itself fails
        logger.critical("CRITICAL FALLBACK: _generate_groups_by_category failed. Attempting very basic random word selection.")
        # (A very rudimentary fallback - not ideal, but better than crashing if possible)
        all_glob_words = _get_all_available_words()
        if len(all_glob_words) < num_words_needed:
            raise ValueError("Not enough global words for absolute fallback.")
        
        solution_groups, all_words_list, group_descriptions, difficulty_indices = {}, [], {}, {}
        selected_fallback_words = random.sample(all_glob_words, num_words_needed)
        for i in range(NUM_GROUPS):
            group_id = f"fallback_group_{i+1}"
            start_idx = i * WORDS_PER_GROUP
            group_words = selected_fallback_words[start_idx : start_idx + WORDS_PER_GROUP]
            solution_groups[group_id] = sorted(group_words)
            all_words_list.extend(group_words)
            group_descriptions[group_id] = f"Random Group {i+1}"
            difficulty_indices[group_id] = i
        return solution_groups, all_words_list, group_descriptions, difficulty_indices


# --- ML Prediction Helper ---
def get_fallback_prediction(params: Dict[str, Any]) -> Dict[str, Any]:
    logger.warning("[FALLBACK] Using rule-based difficulty estimation due to ML issue or deliberate choice.")
    complexity = connection_types.get(params.get('connection_type', 'same_category'), 1)
    rarity = word_rarity_levels.get(params.get('word_rarity', 'common'), 2)
    score = complexity + rarity 
    est_time = 20 + score * 4 + (WORDS_PER_GROUP * NUM_GROUPS - 16) * 0.5
    difficulty = "medium"
    if est_time <= 35: difficulty = "easy"
    elif est_time <= 75: difficulty = "medium"
    else: difficulty = "hard"
    logger.info(f"[FALLBACK] Params: {params}, Score: {score}, Est.Time: {est_time:.1f}s, FallbackDiff: {difficulty}")
    return {'predicted_solve_time': round(est_time, 2), 'difficulty': difficulty, 'is_fallback': True}

def predict_difficulty_for_params(params: Dict[str, Any]) -> Dict[str, Any]:
    if not model_pipeline or not ALL_EXPECTED_INPUT_FEATURES:
        return get_fallback_prediction(params)
    try:
        feature_data_dict = {feature_name: 0.0 for feature_name in ALL_EXPECTED_INPUT_FEATURES}
        feature_data_dict.update({
            'num_words': float(params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP)),
            'connection_complexity': float(connection_types.get(params['connection_type'], 5)),
            'word_rarity_value': float(word_rarity_levels.get(params['word_rarity'], 5)),
            'semantic_distance': float(params.get('semantic_distance', 5.0)),
            'time_of_day': float(datetime.datetime.now().hour),
            'hints_used': 0.0, 
            'num_players': 50.0, 'completions': 40.0, 'completion_rate': 0.80,
            'attempt_count': 2.0, 'time_before_first_attempt': 10.0,
            'hover_count': float(params.get('num_words', 16) * 1.5),
            'abandonment_rate': 0.20, 'competitiveness_score': 5.0, 'frustration_score': 3.0,
            'satisfaction_score': 7.0, 'learning_value': 5.0, 'engagement_score': 6.0,
            'replayability_score': 4.0, 'avg_attempts_before_success': 1.5
        })
        
        final_feature_values = [feature_data_dict.get(name, 0.0) for name in ALL_EXPECTED_INPUT_FEATURES]
        predict_df = pd.DataFrame([final_feature_values], columns=ALL_EXPECTED_INPUT_FEATURES)
        
        predicted_time_array = model_pipeline.predict(predict_df)
        predicted_time = float(predicted_time_array[0])

        difficulty = "medium"
        if predicted_time < 40: difficulty = "easy" 
        elif predicted_time < 80: difficulty = "medium"
        else: difficulty = "hard"
        
        logger.info(f"ML Prediction: Time={predicted_time:.2f}s, Diff='{difficulty}' for params: {params}")
        return {'predicted_solve_time': round(predicted_time, 2), 'difficulty': difficulty, 'is_fallback': False}

    except Exception as e:
        logger.error(f"Error during ML prediction: {e}. Using fallback. Params: {params}", exc_info=True)
        return get_fallback_prediction(params)

# --- Main Puzzle Generation Logic ---
def generate_solvable_puzzle(target_difficulty: str) -> Dict[str, Any]:
    puzzle_id = str(uuid.uuid4())
    num_words_total = NUM_GROUPS * WORDS_PER_GROUP
    
    logger.info(f"\n--- Generating Puzzle for Target Difficulty: '{target_difficulty.upper()}' ---")

    generation_params: Dict[str, Any] = {}
    predicted_result: Dict[str, Any] = {}
    found_matching_params = False
    best_params_so_far = None
    best_prediction_so_far = None

    param_candidates = []
    if target_difficulty == 'easy':
        param_candidates.extend([
            {'connection_type': 'same_category', 'word_rarity': 'common', 'semantic_distance': random.uniform(1,3)},
            {'connection_type': 'same_category', 'word_rarity': 'very_common', 'semantic_distance': random.uniform(1,2)},
            {'connection_type': 'begins_with', 'word_rarity': 'very_common', 'semantic_distance': random.uniform(1,2.5)},
            {'connection_type': 'ends_with', 'word_rarity': 'common', 'semantic_distance': random.uniform(1.5,3.5)},
        ] * (MAX_GENERATION_ATTEMPTS // 3 + 1)) 
    elif target_difficulty == 'medium':
        param_candidates.extend([
            {'connection_type': 'same_category', 'word_rarity': 'somewhat_common', 'semantic_distance': random.uniform(2,5)},
            {'connection_type': 'syllable_count', 'word_rarity': 'common', 'semantic_distance': random.uniform(3,6)},
            {'connection_type': 'rhyming_words', 'word_rarity': 'somewhat_common', 'semantic_distance': random.uniform(3,5)},
            {'connection_type': 'letter_pattern', 'word_rarity': 'common', 'semantic_distance': random.uniform(4,6)},
        ] * (MAX_GENERATION_ATTEMPTS // 3 + 1))
    else: # hard
        param_candidates.extend([
            {'connection_type': 'conceptual_relation', 'word_rarity': 'uncommon', 'semantic_distance': random.uniform(6,9)},
            {'connection_type': 'anagrams', 'word_rarity': 'rare', 'semantic_distance': random.uniform(5,8)},
            {'connection_type': 'metaphorical_relation', 'word_rarity': 'rare', 'semantic_distance': random.uniform(7,10)},
            {'connection_type': 'multiple_rules', 'word_rarity': 'uncommon', 'semantic_distance': random.uniform(5,8)},
        ] * (MAX_GENERATION_ATTEMPTS // 3 + 1))
    random.shuffle(param_candidates)

    for attempt in range(min(MAX_GENERATION_ATTEMPTS, len(param_candidates))):
        current_params = param_candidates[attempt]
        current_params['num_words'] = num_words_total
        
        current_prediction = predict_difficulty_for_params(current_params)
        
        if not best_prediction_so_far or \
           (not current_prediction.get('is_fallback') and best_prediction_so_far.get('is_fallback', True)) or \
           (current_prediction.get('difficulty') == target_difficulty and not current_prediction.get('is_fallback')):
            best_params_so_far = current_params.copy()
            best_prediction_so_far = current_prediction.copy()

        if current_prediction.get('difficulty') == target_difficulty and not current_prediction.get('is_fallback'):
            generation_params = current_params
            predicted_result = current_prediction
            found_matching_params = True
            logger.info(f"--> Target '{target_difficulty}' MET by ML on attempt {attempt+1} with params: {generation_params}")
            break
    
    if not found_matching_params:
        logger.warning(f"Could not find exact ML match for '{target_difficulty}'. Using best found: {best_params_so_far} (Pred: {best_prediction_so_far})")
        if best_params_so_far and best_prediction_so_far:
            generation_params = best_params_so_far
            predicted_result = best_prediction_so_far
        else:
            logger.error("No valid ML predictions made, using hardcoded default params for generation.")
            generation_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total, 'semantic_distance': 3.0}
            predicted_result = get_fallback_prediction(generation_params)

    final_connection_type = generation_params.get('connection_type', 'same_category')
    logger.info(f"--- Finalizing puzzle. Connection: '{final_connection_type}', Rarity: '{generation_params.get('word_rarity')}' ---")
    
    solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = {}, [], {}, {}
    generation_successful_flag = False
    
    try:
        if final_connection_type == 'same_category':
            # Call the correctly defined _generate_groups_by_category
            category_groups_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set()) # Pass empty set for exclude_words initially
            if category_groups_data:
                solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = category_groups_data
                generation_successful_flag = True if len(solution_groups) == NUM_GROUPS else False
            else:
                logger.warning(f"_generate_groups_by_category returned None for '{final_connection_type}'.")
        
        # TODO: >>> ADD ELIF BLOCKS FOR YOUR OTHER connection_type LOGIC HERE <<<
        # Each should call a specific helper like _generate_groups_by_rhyme(...) etc.
        # and populate the four return variables, then set generation_successful_flag.
        
        if not generation_successful_flag:
            logger.warning(f"Specific logic for '{final_connection_type}' failed or not fully implemented. Using _generate_fallback_groups.")
            solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
                _generate_fallback_groups(num_words_total, final_connection_type)
            if len(solution_groups) != NUM_GROUPS or len(all_words_for_grid) != num_words_total:
                 raise ValueError("Fallback generation did not produce a valid puzzle structure.")

    except Exception as e:
        logger.error(f"Exception during word selection for '{final_connection_type}': {e}. Attempting full fallback.", exc_info=True)
        try:
            solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
                _generate_fallback_groups(num_words_total, "same_category") # Safest fallback
            if len(solution_groups) != NUM_GROUPS or len(all_words_for_grid) != num_words_total:
                 raise ValueError("Full fallback generation did not produce a valid puzzle structure.")
        except Exception as fallback_e:
            logger.critical(f"Full fallback generation also failed: {fallback_e}", exc_info=True)
            raise ValueError("CRITICAL: Failed to generate puzzle words after multiple fallbacks.") from fallback_e

    if len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
        logger.critical(f"Puzzle assembly error. Words: {len(all_words_for_grid)}/{num_words_total}, Groups: {len(solution_groups)}/{NUM_GROUPS}")
        raise ValueError("Generated puzzle is malformed (incorrect counts).")

    random.shuffle(all_words_for_grid)

    active_puzzles[puzzle_id] = {
        "puzzle_id": puzzle_id,
        "words_on_grid": [word.upper() for word in all_words_for_grid],
        "solution": {key: [word.upper() for word in words] for key, words in solution_groups.items()},
        "descriptions": group_descriptions,
        "difficulty": predicted_result.get('difficulty', target_difficulty),
        "predicted_solve_time": predicted_result.get('predicted_solve_time', -1.0),
        "is_fallback_prediction": predicted_result.get('is_fallback', True),
        "parameters": {**generation_params, "difficulty_index_map": difficulty_index_map_for_puzzle},
        "creation_time": datetime.datetime.now(),
        "is_daily": False
    }
    logger.info(f"--- Puzzle {puzzle_id} (Target: {target_difficulty}, Final Gen Diff: {active_puzzles[puzzle_id]['difficulty']}) created. ---")

    return {
        "puzzle_id": puzzle_id,
        "words": active_puzzles[puzzle_id]["words_on_grid"],
        "difficulty": active_puzzles[puzzle_id]["difficulty"]
    }

def get_or_generate_daily_challenge() -> dict:
    cleanup_old_puzzles()
    today_str = datetime.date.today().isoformat()
    daily_puzzle_id = f"daily_{today_str}"
    if daily_puzzle_id not in active_puzzles:
        logger.info(f"Generating new daily challenge for {today_str} (ID: {daily_puzzle_id})")
        original_random_state = random.getstate()
        random.seed(today_str) 
        temp_puzzle_client_data = generate_solvable_puzzle(target_difficulty="medium")
        temp_puzzle_id_from_gen = temp_puzzle_client_data["puzzle_id"]
        if temp_puzzle_id_from_gen in active_puzzles:
            daily_puzzle_server_data = active_puzzles.pop(temp_puzzle_id_from_gen)
            daily_puzzle_server_data["puzzle_id"] = daily_puzzle_id
            daily_puzzle_server_data["is_daily"] = True
            daily_puzzle_server_data["difficulty"] = "Daily Challenge"
            active_puzzles[daily_puzzle_id] = daily_puzzle_server_data
            logger.info(f"Daily challenge {daily_puzzle_id} (based on temp {temp_puzzle_id_from_gen}) stored.")
        else:
            random.setstate(original_random_state); raise ValueError("Daily temp puzzle not found.")
        random.setstate(original_random_state)
    else: logger.info(f"Returning existing daily challenge for {today_str}")
    puzzle_to_send = active_puzzles[daily_puzzle_id]
    return {"puzzle_id": puzzle_to_send["puzzle_id"], "words": puzzle_to_send["words_on_grid"],
            "difficulty": puzzle_to_send["difficulty"], "is_daily": True }

def check_puzzle_answer(puzzle_id: str, user_groups_attempt: Dict[str, List[str]]) -> Dict[str, Any]:
    if puzzle_id not in active_puzzles: return {"correct": False, "message": "Invalid puzzle.", "solved_groups": {}}
    puzzle_data = active_puzzles[puzzle_id]; actual_solution = puzzle_data["solution"]
    actual_descriptions = puzzle_data["descriptions"]; difficulty_index_map = puzzle_data.get("parameters", {}).get("difficulty_index_map", {})
    attempted_words_key = next(iter(user_groups_attempt), None)
    if not attempted_words_key: return {"correct": False, "message": "No attempt.", "solved_groups": {}}
    attempted_words_sorted_upper = sorted([word.upper() for word in user_groups_attempt[attempted_words_key]])
    if len(attempted_words_sorted_upper) != WORDS_PER_GROUP: return {"correct": False, "message": f"Select {WORDS_PER_GROUP} words.", "solved_groups": {}}
    for group_key, correct_words_sorted_upper in actual_solution.items():
        if attempted_words_sorted_upper == correct_words_sorted_upper:
            return {"correct": True, "message": f"Correct! Category: {actual_descriptions.get(group_key, 'Found')}",
                    "solved_groups": { group_key: {"description": actual_descriptions.get(group_key, "Unknown"),
                                      "difficulty_index": difficulty_index_map.get(group_key, 0) }}}
    return {"correct": False, "message": "Incorrect group.", "solved_groups": {}}

def get_puzzle_hint(puzzle_id: str, solved_group_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    if solved_group_keys is None: solved_group_keys = []
    if puzzle_id not in active_puzzles: return {"hint": None, "message": "Invalid puzzle.", "words": []}
    puzzle_data = active_puzzles[puzzle_id]; actual_solution = puzzle_data.get("solution", {})
    actual_descriptions = puzzle_data.get("descriptions", {})
    unsolved_groups = {k: {"words": v, "description": actual_descriptions.get(k, "A Group")} for k,v in actual_solution.items() if k not in solved_group_keys}
    if not unsolved_groups: return {"hint": None, "message": "All groups solved!", "words": []}
    target_group_key = random.choice(list(unsolved_groups.keys())); target_group_info = unsolved_groups[target_group_key]
    words_for_hint_highlight = []; hint_text_core = f"Consider: '{target_group_info['description']}'."
    target_group_words = target_group_info.get("words", [])
    if len(target_group_words) >= 2: words_for_hint_highlight = random.sample(target_group_words, 2)
    elif target_group_words: words_for_hint_highlight = random.sample(target_group_words, 1)
    
    if words_for_hint_highlight: hint_text_core += f" Words like {words_for_hint_highlight[0]}"
    if len(words_for_hint_highlight) > 1: hint_text_core += f" and {words_for_hint_highlight[1]}"
    if words_for_hint_highlight: hint_text_core += " might belong."
    
    return {"hint": hint_text_core, "words": words_for_hint_highlight, "message": "Hint provided."}

def cleanup_old_puzzles():
    current_time_dt = datetime.datetime.now(); today_date = datetime.date.today()
    puzzles_to_delete = []
    for pid, data in list(active_puzzles.items()):
        creation_time_dt = data.get("creation_time")
        if not isinstance(creation_time_dt, datetime.datetime):
            if isinstance(creation_time_dt, float): creation_time_dt = datetime.datetime.fromtimestamp(creation_time_dt)
            else: logger.warning(f"Puzzle {pid} invalid creation_time: {creation_time_dt}. Skip cleanup."); continue
        if pid.startswith("daily_"):
            try:
                if datetime.date.fromisoformat(pid.replace("daily_", "")) < today_date: puzzles_to_delete.append(pid)
            except ValueError: logger.error(f"Malformed daily puzzle ID for cleanup: {pid}")
        elif (current_time_dt - creation_time_dt).total_seconds() > MAX_PUZZLE_AGE_SECONDS:
            puzzles_to_delete.append(pid)
    for pid in puzzles_to_delete:
        if pid in active_puzzles: del active_puzzles[pid]; logger.info(f"Cleaned up puzzle: {pid}")

if __name__ == "__main__":
    logger.info("Testing puzzle_logic.py standalone functions...")
    for diff_test in ["easy", "medium", "hard"]:
        try:
            print(f"\n--- Testing Regular Puzzle Generation ({diff_test.capitalize()}) ---")
            puzzle = generate_solvable_puzzle(target_difficulty=diff_test)
            print(f"{diff_test.capitalize()} Puzzle Client Data: {puzzle}")
            if puzzle['puzzle_id'] in active_puzzles:
                print(f"  Server Data: Difficulty='{active_puzzles[puzzle['puzzle_id']]['difficulty']}', PredTime={active_puzzles[puzzle['puzzle_id']]['predicted_solve_time']:.1f}s, IsFallbackPred={active_puzzles[puzzle['puzzle_id']]['is_fallback_prediction']}")
                print(f"  Server Params Used: {active_puzzles[puzzle['puzzle_id']]['parameters']}")
        except Exception as e:
            print(f"Error generating {diff_test} puzzle: {e}", exc_info=True)
    try:
        print("\n--- Testing Daily Challenge Generation ---")
        daily_data_client = get_or_generate_daily_challenge()
        print(f"Daily Challenge Client Data: {daily_data_client}")
    except Exception as e:
        print(f"Error with daily challenge: {e}", exc_info=True)
    logger.info("puzzle_logic.py tests completed.")