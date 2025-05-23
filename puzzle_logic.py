import random
import uuid
import joblib # For loading scikit-learn models/pipelines
import pandas as pd
import os
import datetime
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Set

# Gensim import for Word2Vec
try:
    from gensim.models import Word2Vec, KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    KeyedVectors = None # Define for type hinting if gensim not available
    Word2Vec = None     # Define for type hinting

# --- Initialize Logger ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if this module is reloaded
    # BasicConfig should ideally be called only once at the application entry point (e.g., main.py)
    # However, for standalone testing of this module, it's included here.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

# --- Constants ---
MODEL_DIR = "model"
# !!! IMPORTANT: Set this to the core name of your trained difficulty prediction model file !!!
# Example: if file is "wordlinks_ridge_regression_model.pkl", then BEST_MODEL_NAME_FROM_TRAINING = "ridge_regression"
BEST_MODEL_NAME_FROM_TRAINING = "ridge_regression" # <--- !!! UPDATE THIS !!!
DIFFICULTY_MODEL_FILENAME = os.path.join(MODEL_DIR, f"wordlinks_{BEST_MODEL_NAME_FROM_TRAINING}_model.pkl")
FEATURE_LIST_FILENAME = os.path.join(MODEL_DIR, "feature_list.pkl")

# !!! UPDATE THIS TO THE EXACT FILENAME OF YOUR W2V MODEL IN THE 'model/' FOLDER !!!
WORD2VEC_ACTUAL_FILENAME = "glove.6B.100d.kv" # <--- EXAMPLE, CHANGE THIS!
WORD2VEC_MODEL_PATH = os.path.join(MODEL_DIR, WORD2VEC_ACTUAL_FILENAME)

MAX_GENERATION_ATTEMPTS = 30 
WORDS_PER_GROUP = 4
NUM_GROUPS = 4
MAX_PUZZLE_AGE_SECONDS = 3 * 3600 # 3 hours
MAX_HINTS = 3 # For hint logic and personalization defaults

# --- Load Difficulty Prediction ML Model Pipeline ---
model_pipeline = None
ALL_EXPECTED_INPUT_FEATURES = [] # Order in this list must match training
TRAINING_NUMERIC_FEATURES = []   # Loaded from feature_list.pkl
TRAINING_CATEGORICAL_FEATURES = [] # Loaded from feature_list.pkl

try:
    if not os.path.isdir(MODEL_DIR):
        logger.warning(f"Model directory '{MODEL_DIR}' does not exist. ML for difficulty disabled.")
    elif not os.path.exists(DIFFICULTY_MODEL_FILENAME):
        logger.warning(f"Difficulty model file '{DIFFICULTY_MODEL_FILENAME}' not found. ML for difficulty disabled.")
    elif not os.path.exists(FEATURE_LIST_FILENAME):
        logger.warning(f"Feature list file '{FEATURE_LIST_FILENAME}' not found. ML for difficulty disabled.")
    else:
        model_pipeline = joblib.load(DIFFICULTY_MODEL_FILENAME)
        feature_info = joblib.load(FEATURE_LIST_FILENAME)
        
        TRAINING_NUMERIC_FEATURES = feature_info.get('numeric_features', [])
        TRAINING_CATEGORICAL_FEATURES = feature_info.get('categorical_features', [])
        # Crucial: The order of features here must match how the preprocessor in your pipeline expects them.
        # Typically, this order is derived from how X_train.columns was when you trained.
        ALL_EXPECTED_INPUT_FEATURES = TRAINING_NUMERIC_FEATURES + TRAINING_CATEGORICAL_FEATURES 

        if not ALL_EXPECTED_INPUT_FEATURES:
            logger.error("Feature list from '{FEATURE_LIST_FILENAME}' is empty or malformed. ML for difficulty disabled.")
            model_pipeline = None
        else:
            logger.info(f"Difficulty Prediction ML Model ('{DIFFICULTY_MODEL_FILENAME}') loaded successfully.")
            logger.debug(f"Expected input features for difficulty model (order matters): {ALL_EXPECTED_INPUT_FEATURES}")
except Exception as e:
    logger.error(f"Error loading Difficulty Prediction ML model or feature list: {e}. ML for difficulty disabled.", exc_info=True)
    model_pipeline = None
    ALL_EXPECTED_INPUT_FEATURES = []


# --- Load Word2Vec Model ---
w2v_model = None
if GENSIM_AVAILABLE:
    try:
        if not os.path.exists(WORD2VEC_MODEL_PATH):
            logger.warning(f"Word2Vec model file '{WORD2VEC_MODEL_PATH}' not found in '{MODEL_DIR}'. Word2Vec hints will use basic fallback.")
        else:
            logger.info(f"Attempting to load Word2Vec model from: {WORD2VEC_MODEL_PATH} (This may take time for large models)...")
            # --- !!! CHOOSE AND UNCOMMENT THE CORRECT LOADING METHOD FOR YOUR W2V FILE !!! ---
            # Option 1: For .kv files (e.g., GloVe converted and saved with model.save())
            w2v_model = KeyedVectors.load(WORD2VEC_MODEL_PATH)
            
            # Option 2: For .bin files (like GoogleNews-vectors-negative300.bin)
            # w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)
            # To load only a subset of vectors to save memory (e.g., top 500k):
            # w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True, limit=500000)

            # Option 3: For models saved with gensim's full model.save() (e.g., custom_word2vec.model)
            # w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH) 
            # Then you would typically access vectors via w2v_model.wv (e.g., w2v_model.wv.most_similar)

            # Option 4: For text-based vector files (.txt, .vec)
            # w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=False) # Add no_header=True if needed for your specific format
            
            logger.info(f"Word2Vec model '{WORD2VEC_MODEL_PATH}' loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Word2Vec model from '{WORD2VEC_MODEL_PATH}': {e}. Word2Vec hints will use basic fallback.", exc_info=True)
        w2v_model = None
else:
    logger.warning("Gensim library not installed/found. Word2Vec-based hints will use basic fallback.")


# --- Data Definitions (Expand with your own rich dataset) ---
word_categories = {
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'brown', 'gray', 'teal', 'maroon', 'indigo', 'violet', 'cyan', 'magenta', 'lime', 'olive', 'gold', 'silver', 'bronze', 'ivory', 'beige'],
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'snake', 'eagle', 'shark', 'whale', 'dolphin', 'penguin', 'owl', 'bat', 'bee', 'ant'],
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach', 'kiwi', 'watermelon', 'cherry', 'pear', 'blueberry', 'raspberry', 'lemon', 'lime', 'plum', 'apricot', 'fig', 'date', 'guava'],
    # Add many more categories and words for variety
    'kitchen_items': ['knife', 'fork', 'spoon', 'plate', 'bowl', 'cup', 'pan', 'pot', 'oven', 'microwave', 'blender', 'toaster', 'kettle', 'grater', 'whisk', 'ladle', 'spatula', 'colander']
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
word_rarity_levels = { # You might need a way to assign these to your actual words
    'very_common': 1, 'common': 2, 'somewhat_common': 4, 'uncommon': 6,
    'rare': 8, 'very_rare': 10, 'extremely_rare': 12
}

# In-memory storage for active puzzles
active_puzzles: Dict[str, Dict[str, Any]] = {}


# --- Helper Functions ---
def _get_all_available_words(exclude_words: Optional[Set[str]] = None) -> List[str]:
    if exclude_words is None: exclude_words = set()
    all_words = set()
    for cat_words_list in word_categories.values():
        for word in cat_words_list:
            all_words.add(word.lower())
    return list(all_words - exclude_words)

def _generate_groups_by_category(num_groups_to_gen: int, words_per_group_val: int, used_words_global: Set[str]) -> Optional[Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]]:
    """Generates groups based on 'same_category', ensuring words are not reused globally FOR THIS PUZZLE GENERATION."""
    if len(word_categories) < num_groups_to_gen:
        logger.warning(f"CatGen: Not enough categories ({len(word_categories)}) for {num_groups_to_gen} groups.")
        return None
    
    # Filter categories by those that have enough words *not already in used_words_global*
    available_categories_for_selection = {}
    for cat, words in word_categories.items():
        potential_words = [w.lower() for w in words if w.lower() not in used_words_global]
        if len(potential_words) >= words_per_group_val:
            available_categories_for_selection[cat] = potential_words
            
    if len(available_categories_for_selection) < num_groups_to_gen:
        logger.warning(f"CatGen: Not enough valid categories with sufficient unique global words. Found {len(available_categories_for_selection)}, need {num_groups_to_gen}.")
        return None
        
    selected_cat_names = random.sample(list(available_categories_for_selection.keys()), num_groups_to_gen)
    
    solution_groups, all_words_for_this_puzzle, descriptions, diff_indices = {}, [], {}, {}
    
    for i, cat_name in enumerate(selected_cat_names):
        group_id = f"group_{i+1}"
        # Words from this category, already filtered against used_words_global
        words_for_this_group = available_categories_for_selection[cat_name]
        
        # We still need to sample *from these available words* without replacement for this puzzle
        # This check should have been covered by the initial filter, but as a safeguard:
        if len(words_for_this_group) < words_per_group_val:
             logger.error(f"CatGen: INTERNAL ERROR - Category '{cat_name}' should have enough words but doesn't. Has {len(words_for_this_group)}.")
             return None # Should ideally not happen if initial filtering is correct.

        group_words = random.sample(words_for_this_group, words_per_group_val)
        
        solution_groups[group_id] = sorted(group_words)
        all_words_for_this_puzzle.extend(group_words)
        # No need to update used_words_global here as it's for words already used in *other* puzzles or contexts.
        # We track words used within *this current puzzle* by simply not picking them again from remaining categories if we were to build more complex puzzles.
        descriptions[group_id] = connection_descriptions.get('same_category', "Category") + f": {cat_name.capitalize()}"
        diff_indices[group_id] = i 
    
    # Final check to ensure no duplicate words ended up in all_words_for_this_puzzle (shouldn't happen if logic is correct)
    if len(all_words_for_this_puzzle) != len(set(all_words_for_this_puzzle)):
        logger.error("CatGen: Duplicate words found across generated groups for this puzzle. This indicates an issue in category word lists or selection logic.")
        return None

    return solution_groups, all_words_for_this_puzzle, descriptions, diff_indices


def _generate_fallback_groups(num_words_needed: int, original_connection_type: str) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]:
    logger.warning(f"[FALLBACK-GROUPS] Using 'same_category' logic as fallback for original request '{original_connection_type}'.")
    # Attempt to generate groups using 'same_category' logic.
    # Pass an empty set for used_words_global as this is a fresh attempt for this puzzle.
    category_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set()) 
    if category_data:
        return category_data
    else:
        logger.critical("CRITICAL FALLBACK FAILURE: _generate_groups_by_category also failed during fallback. Generating dummy data.")
        # Create absolute dummy data to prevent a hard crash if all else fails
        dummy_solution = {f"group_{i+1}": [f"fallback_word_{i*WORDS_PER_GROUP + j + 1}" for j in range(WORDS_PER_GROUP)] for i in range(NUM_GROUPS)}
        dummy_words = [w for grp_words in dummy_solution.values() for w in grp_words]
        dummy_descriptions = {k: "Error: Fallback Group" for k in dummy_solution}
        dummy_difficulty_indices = {k: i for i, k in enumerate(dummy_solution)}
        return dummy_solution, dummy_words, dummy_descriptions, dummy_difficulty_indices


def get_fallback_prediction(params: Dict[str, Any]) -> Dict[str, Any]:
    logger.warning("[FALLBACK-PRED] Using rule-based difficulty estimation for ML prediction.")
    complexity = connection_types.get(params.get('connection_type', 'same_category'), 1)
    rarity = word_rarity_levels.get(params.get('word_rarity', 'common'), 2)
    score = complexity + rarity 
    # Adjusted heuristic for time based on typical values
    est_time = 20 + (score * 4) + (params.get('num_words', 16) - 16) * 0.5 
    difficulty = "medium"
    if est_time <= 35: difficulty = "easy"
    elif est_time <= 75: difficulty = "medium"
    else: difficulty = "hard"
    return {'predicted_solve_time': round(est_time, 2), 'difficulty': difficulty, 'is_fallback': True}

def predict_difficulty_for_params(params: Dict[str, Any]) -> Dict[str, Any]:
    if not model_pipeline or not ALL_EXPECTED_INPUT_FEATURES:
        return get_fallback_prediction(params)
    try:
        feature_data_dict = {name: 0.0 for name in ALL_EXPECTED_INPUT_FEATURES} # Initialize all expected features
        feature_data_dict.update({
            'num_words': float(params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP)),
            'connection_complexity': float(connection_types.get(params.get('connection_type'), 5)), # Use get for safety
            'word_rarity_value': float(word_rarity_levels.get(params.get('word_rarity'), 5)), # Use get for safety
            'semantic_distance': float(params.get('semantic_distance', 5.0)),
            'time_of_day': float(datetime.datetime.now().hour),
            'hints_used': 0.0, # For generation
            # Default values for other features your model might expect:
            'num_players': 50.0, 'completions': 40.0, 'completion_rate': 0.80, 
            'attempt_count': 2.0, 'time_before_first_attempt': 10.0, 
            'hover_count': float(params.get('num_words', 16) * 1.5), 
            'abandonment_rate': 0.20, 'competitiveness_score': 5.0, 'frustration_score': 3.0,
            'satisfaction_score': 7.0, 'learning_value': 5.0, 'engagement_score': 6.0,
            'replayability_score': 4.0, 'avg_attempts_before_success': 1.5
        })
        
        # Ensure only expected features are passed and in the correct order
        final_feature_values = []
        for feature_name in ALL_EXPECTED_INPUT_FEATURES:
            final_feature_values.append(feature_data_dict.get(feature_name, 0.0)) # Default to 0.0 if somehow missing after update
            
        if len(final_feature_values) != len(ALL_EXPECTED_INPUT_FEATURES):
            logger.error(f"Feature length mismatch before creating DataFrame! Expected {len(ALL_EXPECTED_INPUT_FEATURES)}, got {len(final_feature_values)}")
            return get_fallback_prediction(params)

        predict_df = pd.DataFrame([final_feature_values], columns=ALL_EXPECTED_INPUT_FEATURES)
        
        predicted_time_array = model_pipeline.predict(predict_df)
        predicted_time = float(predicted_time_array[0])

        difficulty = "medium" # Default thresholding
        if predicted_time < 40: difficulty = "easy"
        elif predicted_time < 80: difficulty = "medium"
        else: difficulty = "hard"
        
        logger.info(f"ML Prediction: SolveTime={predicted_time:.2f}s, Difficulty='{difficulty}' for Params: {params}")
        return {'predicted_solve_time': round(predicted_time, 2), 'difficulty': difficulty, 'is_fallback': False}
    except Exception as e:
        logger.error(f"Error in ML predict_difficulty_for_params: {e}", exc_info=True)
        return get_fallback_prediction(params)

def generate_solvable_puzzle(target_difficulty: str, user_performance_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    puzzle_id = str(uuid.uuid4())
    num_words_total = NUM_GROUPS * WORDS_PER_GROUP
    
    logger.info(f"\n--- Generating Puzzle. Target Diff: '{target_difficulty.upper()}' ---")
    if user_performance_summary:
        logger.info(f"User Performance for '{target_difficulty}': Plays={user_performance_summary.get('plays',0)}, WinRate={user_performance_summary.get('win_rate',0.0):.2f}, AvgHints={user_performance_summary.get('avg_hints', MAX_HINTS / 2.0):.1f}")
    else:
        logger.info("No user performance summary provided for this puzzle generation.")

    generation_params: Dict[str, Any] = {}
    predicted_result: Dict[str, Any] = {}
    found_matching_params = False
    best_params_so_far = None
    best_prediction_so_far = None
    
    param_candidates = []
    actual_search_difficulty = target_difficulty

    if user_performance_summary and user_performance_summary.get("plays", 0) >= 2: 
        win_rate = user_performance_summary.get("win_rate", 0.5) 
        avg_hints = user_performance_summary.get("avg_hints", MAX_HINTS / 2.0)
        if target_difficulty == "easy" and win_rate > 0.9 and avg_hints < 0.2: actual_search_difficulty = "medium"
        elif target_difficulty == "medium":
            if win_rate > 0.75 and avg_hints < 0.5: actual_search_difficulty = "hard"
            elif win_rate < 0.3 and avg_hints > (MAX_HINTS * 0.66): actual_search_difficulty = "easy"
        elif target_difficulty == "hard" and win_rate < 0.25 and avg_hints > (MAX_HINTS * 0.66) : actual_search_difficulty = "medium"
        if actual_search_difficulty != target_difficulty:
            logger.info(f"Personalization: Nudging search from '{target_difficulty}' to '{actual_search_difficulty}' based on performance.")
    
    # Populate param_candidates based on actual_search_difficulty
    if actual_search_difficulty == 'easy':
        param_candidates.extend([
            {'connection_type': 'same_category', 'word_rarity': 'very_common', 'semantic_distance': random.uniform(1,2.5)},
            {'connection_type': 'same_category', 'word_rarity': 'common', 'semantic_distance': random.uniform(1.5,3.5)},
            {'connection_type': 'begins_with', 'word_rarity': 'very_common', 'semantic_distance': random.uniform(1,3)},
        ] * (MAX_GENERATION_ATTEMPTS // 3 + 1)) # Ensure enough candidates
    elif actual_search_difficulty == 'medium':
        param_candidates.extend([
            {'connection_type': 'same_category', 'word_rarity': 'somewhat_common', 'semantic_distance': random.uniform(2.5,5)},
            {'connection_type': 'syllable_count', 'word_rarity': 'common', 'semantic_distance': random.uniform(3,6)},
            {'connection_type': 'rhyming_words', 'word_rarity': 'somewhat_common', 'semantic_distance': random.uniform(3.5,5.5)},
        ] * (MAX_GENERATION_ATTEMPTS // 3 + 1))
    else: # hard
        param_candidates.extend([
            {'connection_type': 'conceptual_relation', 'word_rarity': 'uncommon', 'semantic_distance': random.uniform(5,8)},
            {'connection_type': 'synonym_groups', 'word_rarity': 'rare', 'semantic_distance': random.uniform(6,9)},
            {'connection_type': 'anagrams', 'word_rarity': 'very_rare', 'semantic_distance': random.uniform(5,7)},
        ] * (MAX_GENERATION_ATTEMPTS // 3 + 1))
    if not param_candidates: # Absolute fallback for candidates
         param_candidates = [{'connection_type': 'same_category', 'word_rarity': 'common', 'semantic_distance': 3.0}] * MAX_GENERATION_ATTEMPTS
    random.shuffle(param_candidates)

    for attempt in range(min(MAX_GENERATION_ATTEMPTS, len(param_candidates))):
        current_params = param_candidates[attempt]
        current_params['num_words'] = num_words_total
        if current_params.get('connection_type') not in connection_types or \
           current_params.get('word_rarity') not in word_rarity_levels:
            continue # Skip if params are somehow invalid
        
        current_prediction = predict_difficulty_for_params(current_params)
        
        if not best_prediction_so_far or \
           (not current_prediction.get('is_fallback') and best_prediction_so_far.get('is_fallback', True)) or \
           (current_prediction.get('difficulty') == target_difficulty and not current_prediction.get('is_fallback') and \
            (not best_prediction_so_far.get('is_fallback',True) or best_prediction_so_far.get('difficulty') != target_difficulty ) ) :
            best_params_so_far = current_params.copy()
            best_prediction_so_far = current_prediction.copy()

        if current_prediction.get('difficulty') == target_difficulty and not current_prediction.get('is_fallback'):
            generation_params = current_params
            predicted_result = current_prediction
            found_matching_params = True
            logger.info(f"--> Target '{target_difficulty}' MET by ML on attempt {attempt+1} with params: {generation_params}")
            break
    
    if not found_matching_params:
        logger.warning(f"Could not find exact ML match for '{target_difficulty}'. Using best valid attempt or ultimate fallback.")
        if best_params_so_far and best_prediction_so_far:
            generation_params = best_params_so_far
            predicted_result = best_prediction_so_far
            logger.info(f"--> Using best ML attempt: {generation_params}, Predicted: {predicted_result}")
        else:
            logger.error("No valid ML predictions made; using hardcoded default params for generation.")
            generation_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total, 'semantic_distance': 3.0}
            predicted_result = get_fallback_prediction(generation_params)

    final_connection_type = generation_params.get('connection_type', 'same_category')
    logger.info(f"--- Finalizing puzzle. Connection: '{final_connection_type}', Rarity: '{generation_params.get('word_rarity')}' ---")
    solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = {}, [], {}, {}
    
    word_selection_data = None
    # TODO: Implement robust word selection for EACH connection_type.
    if final_connection_type == 'same_category':
        word_selection_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set())
    # Add other elif final_connection_type == 'your_other_type': ...
    else:
        logger.warning(f"Word selection for '{final_connection_type}' not implemented. Using fallback.")

    if word_selection_data:
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = word_selection_data
    else: 
        logger.info(f"Specific word selection failed or not implemented. Using fallback groups for '{final_connection_type}'.")
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
            _generate_fallback_groups(num_words_total, final_connection_type)

    if len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
        logger.critical(f"CRITICAL: Puzzle assembly failed. Words: {len(all_words_for_grid)}/{num_words_total}, Groups: {len(solution_groups)}/{NUM_GROUPS}")
        # Attempt one last full fallback if critical error
        try:
            logger.warning("Attempting emergency fallback generation due to assembly failure.")
            solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
                _generate_fallback_groups(num_words_total, "same_category") 
            if len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
                 raise ValueError("Emergency fallback also failed to produce a valid puzzle structure.")
        except Exception as emergency_e:
            logger.critical(f"Emergency fallback failed: {emergency_e}")
            raise ValueError("Generated puzzle is malformed even after emergency fallback.") from emergency_e

    random.shuffle(all_words_for_grid)
    active_puzzles[puzzle_id] = {
        "puzzle_id": puzzle_id, "words_on_grid": [w.upper() for w in all_words_for_grid],
        "solution": {k: [w.upper() for w in v] for k,v in solution_groups.items()}, 
        "descriptions": group_descriptions, "difficulty": predicted_result.get('difficulty', target_difficulty),
        "predicted_solve_time": predicted_result.get('predicted_solve_time', -1.0),
        "is_fallback_prediction": predicted_result.get('is_fallback', True),
        "parameters": {**generation_params, "difficulty_index_map": difficulty_index_map_for_puzzle},
        "creation_time": datetime.datetime.now(), "is_daily": False
    }
    logger.info(f"--- Puzzle {puzzle_id} (Target: {target_difficulty}, Final Gen Diff: {active_puzzles[puzzle_id]['difficulty']}) created. ---")
    return {"puzzle_id": puzzle_id, "words": active_puzzles[puzzle_id]["words_on_grid"], "difficulty": active_puzzles[puzzle_id]["difficulty"]}

def get_or_generate_daily_challenge() -> dict:
    cleanup_old_puzzles(); today_str = datetime.date.today().isoformat(); daily_puzzle_id = f"daily_{today_str}"
    if daily_puzzle_id not in active_puzzles:
        logger.info(f"Generating new daily for {today_str}"); original_random_state = random.getstate(); random.seed(today_str)
        temp_data = generate_solvable_puzzle(target_difficulty="medium") # Daily does not use user_performance_summary
        temp_id = temp_data["puzzle_id"]
        if temp_id in active_puzzles:
            daily_details = active_puzzles.pop(temp_id)
            daily_details.update({"puzzle_id": daily_puzzle_id, "is_daily": True, "difficulty": "Daily Challenge"})
            active_puzzles[daily_puzzle_id] = daily_details
        else: random.setstate(original_random_state); raise ValueError("Daily temp puzzle error.")
        random.setstate(original_random_state)
    else: logger.info(f"Returning existing daily challenge for {today_str}")
    puzzle = active_puzzles[daily_puzzle_id]
    return {"puzzle_id": puzzle["puzzle_id"], "words": puzzle["words_on_grid"],
            "difficulty": puzzle["difficulty"], "is_daily": True }

def check_puzzle_answer(puzzle_id: str, user_groups_attempt: Dict[str, List[str]]) -> Dict[str, Any]:
    if puzzle_id not in active_puzzles: return {"correct": False, "message": "Invalid puzzle.", "solved_groups": {}}
    data = active_puzzles[puzzle_id]; solution = data["solution"]; descriptions = data["descriptions"]
    diff_idx_map = data.get("parameters", {}).get("difficulty_index_map", {})
    attempt_key = next(iter(user_groups_attempt), None)
    if not attempt_key: return {"correct": False, "message": "No attempt.", "solved_groups": {}}
    attempt_words = sorted([w.upper() for w in user_groups_attempt[attempt_key]])
    if len(attempt_words) != WORDS_PER_GROUP: return {"correct": False, "message": f"Select {WORDS_PER_GROUP} words.", "solved_groups": {}}
    for gk, correct_words in solution.items():
        if attempt_words == correct_words:
            # TODO: Mark group gk as solved in active_puzzles[puzzle_id]['_solved_server_side_groups'].add(gk)
            # This helps prevent re-solving and can inform smarter hint generation.
            return {"correct": True, "message": f"Correct! Category: {descriptions.get(gk, 'Found')}",
                    "solved_groups": {gk: {"description": descriptions.get(gk, "Unknown"), "difficulty_index": diff_idx_map.get(gk,0)}}}
    return {"correct": False, "message": "Incorrect group.", "solved_groups": {}}

def get_puzzle_hint(puzzle_id: str, solved_group_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    if solved_group_keys is None: solved_group_keys = []
    if puzzle_id not in active_puzzles:
        return {"hint": None, "message": "Invalid or expired puzzle.", "words": []}
    puzzle_data = active_puzzles[puzzle_id]
    actual_solution = puzzle_data.get("solution", {})
    actual_descriptions = puzzle_data.get("descriptions", {})
    all_words_on_grid_upper = set(puzzle_data.get("words_on_grid", []))

    unsolved_groups = {k: {"words_upper": v, "words_lower": [w.lower() for w in v], 
                           "description": actual_descriptions.get(k, "A Group")}
                       for k, v in actual_solution.items() if k not in solved_group_keys}
    if not unsolved_groups: return {"hint": None, "message": "All groups solved!", "words": []}
    
    target_group_key = random.choice(list(unsolved_groups.keys()))
    target_group_info = unsolved_groups[target_group_key]
    hint_text = f"Hint for '{target_group_info['description']}'."
    words_to_highlight = [] # Should be UPPERCASE

    if w2v_model and GENSIM_AVAILABLE:
        try:
            anchor_words_lower = random.sample(target_group_info["words_lower"], k=min(len(target_group_info["words_lower"]), 1))
            # Use w2v_model directly if it's KeyedVectors, or w2v_model.wv if it's a full Word2Vec model
            vocab_source = w2v_model.wv if hasattr(w2v_model, 'wv') and hasattr(w2v_model.wv, 'key_to_index') else w2v_model
            
            valid_anchor_words = [w for w in anchor_words_lower if w in vocab_source] # Check against appropriate vocab
            
            if valid_anchor_words:
                similar_candidates = vocab_source.most_similar(positive=valid_anchor_words, topn=15)
                found_w2v_hint_word_upper = None
                for sim_word_lower, _ in similar_candidates:
                    sim_word_upper = sim_word_lower.upper()
                    is_in_any_solution = any(sim_word_upper in sol_group for sol_group in actual_solution.values())
                    if sim_word_lower.isalpha() and len(sim_word_lower) > 2 and \
                       sim_word_upper not in all_words_on_grid_upper and not is_in_any_solution:
                        found_w2v_hint_word_upper = sim_word_upper; break
                if found_w2v_hint_word_upper:
                    hint_text = f"This group relates to '{target_group_info['description']}'. Think about concepts around '{valid_anchor_words[0].upper()}' or ideas like '{found_w2v_hint_word_upper}' (though '{found_w2v_hint_word_upper}' itself is not in this puzzle)."
                    words_to_highlight.append(valid_anchor_words[0].upper()) # Highlight an original word
                else: logger.info("W2V found no suitable distinct hint word for group '%s'. Fallback hint.", target_group_key)
            else: logger.info("Anchor words for W2V hint not in vocab for group '%s'. Fallback hint.", target_group_key)
        except Exception as e: logger.error(f"Error in W2V hint gen for group '%s': {e}", target_group_key, exc_info=True) # Fallback handled below
    
    if not words_to_highlight: # Fallback if W2V failed or didn't produce highlightable words
        logger.info("Using basic hint (revealing words from group '%s').", target_group_key)
        if len(target_group_info["words_upper"]) >= 2: words_to_highlight = random.sample(target_group_info["words_upper"], 2)
        elif target_group_info["words_upper"]: words_to_highlight = random.sample(target_group_info["words_upper"], 1)
        
        if words_to_highlight: hint_text = f"Hint for '{target_group_info['description']}': Includes {', '.join(words_to_highlight)}."
        else: hint_text = f"Try to find words related to '{target_group_info['description']}'."

    return {"hint": hint_text, "words": words_to_highlight, "message": "Hint provided."}

def cleanup_old_puzzles():
    current_time_dt = datetime.datetime.now(); today_date = datetime.date.today()
    puzzles_to_delete = []
    for pid, data in list(active_puzzles.items()):
        creation_time_dt = data.get("creation_time")
        if not isinstance(creation_time_dt, datetime.datetime):
            if isinstance(creation_time_dt, float): creation_time_dt = datetime.datetime.fromtimestamp(creation_time_dt)
            else: logger.warning(f"Puzzle {pid} invalid creation_time. Skip cleanup."); continue
        if pid.startswith("daily_"):
            try:
                if datetime.date.fromisoformat(pid.replace("daily_", "")) < today_date: puzzles_to_delete.append(pid)
            except ValueError: logger.error(f"Malformed daily ID for cleanup: {pid}")
        elif (current_time_dt - creation_time_dt).total_seconds() > MAX_PUZZLE_AGE_SECONDS:
            puzzles_to_delete.append(pid)
    for pid in puzzles_to_delete:
        if pid in active_puzzles: del active_puzzles[pid]; logger.info(f"Cleaned up puzzle: {pid}")

if __name__ == "__main__":
    logger.info("Testing puzzle_logic.py standalone functions...")
    mock_good_perf_medium = {"plays": 5, "win_rate": 0.9, "avg_hints": 0.1}
    mock_struggling_perf_medium = {"plays": 5, "win_rate": 0.2, "avg_hints": 2.5}

    print("\n--- Testing Personalized Puzzle Generation (Medium - Good Perf) ---")
    try:
        puzzle_good_perf = generate_solvable_puzzle(target_difficulty="medium", user_performance_summary=mock_good_perf_medium)
        print(f"Good Perf Medium Puzzle Client Data: {puzzle_good_perf}")
        if puzzle_good_perf and puzzle_good_perf.get('puzzle_id') in active_puzzles:
             print(f"  Server Data: GenParams={active_puzzles[puzzle_good_perf['puzzle_id']]['parameters']}")
    except Exception as e: print(f"Error: {e}", exc_info=True)
    
    print("\n--- Testing Personalized Puzzle Generation (Medium - Struggling Perf) ---")
    try:
        puzzle_struggling_perf = generate_solvable_puzzle(target_difficulty="medium", user_performance_summary=mock_struggling_perf_medium)
        print(f"Struggling Perf Medium Puzzle Client Data: {puzzle_struggling_perf}")
        if puzzle_struggling_perf and puzzle_struggling_perf.get('puzzle_id') in active_puzzles:
             print(f"  Server Data: GenParams={active_puzzles[puzzle_struggling_perf['puzzle_id']]['parameters']}")
    except Exception as e: print(f"Error: {e}", exc_info=True)

    print("\n--- Testing Daily Challenge ---")
    try:
        daily_data = get_or_generate_daily_challenge()
        print(f"Daily Client Data: {daily_data}")
    except Exception as e: print(f"Error with daily challenge: {e}", exc_info=True)
    logger.info("puzzle_logic.py tests completed.")