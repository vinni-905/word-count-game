import random
import uuid
import joblib
import pandas as pd
import os
import datetime
import time
import logging
import json
from typing import Dict, Any, Optional, List, Tuple, Set

# Gensim import for Word2Vec
try:
    from gensim.models import Word2Vec, KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    KeyedVectors = None
    Word2Vec = None

# --- Initialize Logger ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

# --- Constants ---
MODEL_DIR = "model"
# Updated based on your 'dir' output
BEST_MODEL_NAME_FROM_TRAINING = "ridge_regression"
DIFFICULTY_MODEL_FILENAME = os.path.join(MODEL_DIR, f"wordlinks_{BEST_MODEL_NAME_FROM_TRAINING}_model.pkl")
FEATURE_LIST_FILENAME = os.path.join(MODEL_DIR, "feature_list.pkl")

# Updated based on your 'dir' output
WORD2VEC_ACTUAL_FILENAME = "glove.6B.100d.kv"
WORD2VEC_MODEL_PATH = os.path.join(MODEL_DIR, WORD2VEC_ACTUAL_FILENAME)

ENRICHED_VOCAB_PATH = os.path.join(MODEL_DIR, "enriched_word_data.json")

MAX_GENERATION_ATTEMPTS = 30
WORDS_PER_GROUP = 4
NUM_GROUPS = 4
MAX_PUZZLE_AGE_SECONDS = 3 * 3600
MAX_HINTS = 3

# ****** ADJUSTED Difficulty Thresholds - TUNE THESE BASED ON YOUR MODEL'S OUTPUT ******
EASY_THRESHOLD = 110  # Example: Times below this might be 'easy'
MEDIUM_THRESHOLD = 170 # Example: Times between EASY_THRESHOLD and this might be 'medium'
# Times above MEDIUM_THRESHOLD will be 'hard'
# ****** END: Difficulty Thresholds ******

# --- Load Difficulty Prediction ML Model Pipeline ---
model_pipeline = None
ALL_EXPECTED_INPUT_FEATURES = []
try:
    if not BEST_MODEL_NAME_FROM_TRAINING: # Added this check
        logger.warning(f"BEST_MODEL_NAME_FROM_TRAINING is not set. ML for difficulty disabled.")
        model_pipeline = None
    elif not os.path.isdir(MODEL_DIR): logger.warning(f"Model directory '{MODEL_DIR}' does not exist. ML for difficulty disabled.")
    elif not os.path.exists(DIFFICULTY_MODEL_FILENAME): logger.warning(f"Difficulty model '{DIFFICULTY_MODEL_FILENAME}' not found. ML disabled.")
    elif not os.path.exists(FEATURE_LIST_FILENAME): logger.warning(f"Feature list '{FEATURE_LIST_FILENAME}' not found. ML disabled.")
    else:
        model_pipeline = joblib.load(DIFFICULTY_MODEL_FILENAME)
        feature_info = joblib.load(FEATURE_LIST_FILENAME)
        numeric_features = feature_info.get('numeric_features', [])
        categorical_features = feature_info.get('categorical_features', [])
        ALL_EXPECTED_INPUT_FEATURES = numeric_features + categorical_features
        if not ALL_EXPECTED_INPUT_FEATURES:
            logger.error(f"Feature list from '{FEATURE_LIST_FILENAME}' is empty. ML for difficulty disabled.")
            model_pipeline = None
        else:
            logger.info(f"Difficulty Prediction ML Model ('{DIFFICULTY_MODEL_FILENAME}') loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Difficulty Prediction ML model: {e}. ML for difficulty disabled.", exc_info=True)
    model_pipeline = None; ALL_EXPECTED_INPUT_FEATURES = []

# --- Load Word2Vec Model ---
w2v_model = None
if GENSIM_AVAILABLE:
    try:
        if not WORD2VEC_ACTUAL_FILENAME: # Added this check
            logger.warning(f"WORD2VEC_ACTUAL_FILENAME is not set. W2V hints fallback.")
        elif not os.path.exists(WORD2VEC_MODEL_PATH): logger.warning(f"Word2Vec model '{WORD2VEC_MODEL_PATH}' not found. W2V hints fallback.")
        else:
            logger.info(f"Loading Word2Vec model: {WORD2VEC_MODEL_PATH}...")
            # --- CHOOSE THE CORRECT LOADING METHOD FOR YOUR FILE TYPE ---
            if WORD2VEC_ACTUAL_FILENAME.endswith(".kv"):
                 w2v_model = KeyedVectors.load(WORD2VEC_MODEL_PATH) # For .kv (KeyedVectors) files
            elif WORD2VEC_ACTUAL_FILENAME.endswith(".bin") or WORD2VEC_ACTUAL_FILENAME.endswith(".wordvectors"):
                 w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=WORD2VEC_ACTUAL_FILENAME.endswith(".bin")) # For .bin or .wordvectors
            elif WORD2VEC_ACTUAL_FILENAME.endswith(".model"):
                 w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH).wv # For full .model files, access .wv for KeyedVectors
            else: # Fallback for other text-based formats, might need adjustment
                 w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=False)
            logger.info(f"Word2Vec model '{WORD2VEC_MODEL_PATH}' loaded.")
    except Exception as e: logger.error(f"Err loading W2V model: {e}. W2V hints fallback.", exc_info=True); w2v_model = None
else: logger.warning("Gensim not installed. W2V hints fallback.")

# --- Load Enriched Vocabulary ---
ENRICHED_VOCABULARY: Dict[str, Dict[str, Any]] = {}
word_categories_hardcoded = {
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'brown', 'gray', 'teal', 'maroon', 'indigo', 'violet', 'cyan', 'magenta', 'lime', 'olive', 'gold', 'silver', 'bronze', 'ivory', 'beige'],
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'snake', 'eagle', 'shark', 'whale', 'dolphin', 'penguin', 'owl', 'bat', 'bee', 'ant'],
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach', 'kiwi', 'watermelon', 'cherry', 'pear', 'blueberry', 'raspberry', 'lemon', 'lime', 'plum', 'apricot', 'fig', 'date', 'guava'],
    'kitchen_items': ['knife', 'fork', 'spoon', 'plate', 'bowl', 'cup', 'pan', 'pot', 'oven', 'microwave', 'blender', 'toaster', 'kettle', 'grater', 'whisk', 'ladle', 'spatula', 'colander']
}
word_categories: Dict[str, List[str]] = {} # This will be populated from ENRICHED_VOCAB_PATH or fallback
if os.path.exists(ENRICHED_VOCAB_PATH):
    try:
        with open(ENRICHED_VOCAB_PATH, 'r', encoding='utf-8') as f: ENRICHED_VOCABULARY = json.load(f)
        logger.info(f"Enriched vocabulary loaded from {ENRICHED_VOCAB_PATH} ({len(ENRICHED_VOCABULARY)} words).")
        temp_word_categories = {}
        for word, data in ENRICHED_VOCABULARY.items():
            category = data.get("category", "unknown_in_enriched")
            if category not in temp_word_categories: temp_word_categories[category] = []
            temp_word_categories[category].append(word)

        # Prioritize categories from enriched data if valid and complete
        if temp_word_categories and "unknown_in_enriched" not in temp_word_categories and \
           all(len(lst) > 0 for lst in temp_word_categories.values()): # Ensure categories are not empty
            word_categories = temp_word_categories
            logger.info(f"Word categories populated from enriched data ({len(word_categories)} categories).")
        else:
            if not temp_word_categories or "unknown_in_enriched" in temp_word_categories:
                 logger.warning("Enriched vocabulary categories are incomplete or contain 'unknown'; falling back to hardcoded.")
            else: # Some categories might be empty
                 logger.warning("Some categories from enriched vocabulary are empty; falling back to hardcoded.")
            word_categories = word_categories_hardcoded # Fallback to hardcoded
    except Exception as e:
        logger.error(f"Error loading or processing {ENRICHED_VOCAB_PATH}: {e}", exc_info=True)
        word_categories = word_categories_hardcoded # Fallback
else:
    logger.warning(f"'{ENRICHED_VOCAB_PATH}' not found. Using hardcoded categories. Run vocabulary_analyzer.py.")
    word_categories = word_categories_hardcoded

connection_types = { 'same_category': 1, 'begins_with': 2, 'ends_with': 3, 'syllable_count': 4, 'synonym_groups': 5, 'antonym_groups': 6, 'compound_words': 7, 'rhyming_words': 7, 'conceptual_relation': 8, 'multiple_rules': 10, 'letter_pattern': 5, 'anagrams': 9, 'homophones': 6, 'contains_substring': 4, 'metaphorical_relation': 9 }
connection_descriptions = { 'same_category': "Same Category", 'begins_with': "Begin With Same Letter", 'ends_with': "End With Same Letter", 'syllable_count': "Same Syllable Count", 'synonym_groups': "Synonyms", 'antonym_groups': "Antonyms", 'compound_words': "Compound Words", 'rhyming_words': "Rhyming Words", 'conceptual_relation': "Conceptual Link", 'multiple_rules': "Multiple Connections", 'letter_pattern': "Shared Letters", 'anagrams': "Anagrams", 'homophones': "Homophones", 'contains_substring': "Contain Substring", 'metaphorical_relation': "Metaphorical Link"}
word_rarity_levels = { 'very_common': 1, 'common': 2, 'somewhat_common': 4, 'uncommon': 6, 'rare': 8, 'very_rare': 10, 'extremely_rare': 12 }
active_puzzles: Dict[str, Dict[str, Any]] = {}

def _get_all_available_words(exclude_words: Optional[Set[str]] = None) -> List[str]:
    if exclude_words is None: exclude_words = set()
    all_words = set()
    for cat_words_list in word_categories.values(): # Use the populated word_categories
        for word in cat_words_list: all_words.add(word.lower())
    return list(all_words - exclude_words)

def _get_words_by_target_rarity(category_name: str, target_rarity_name: str, count: int, exclude_list: Set[str]) -> List[str]:
    if not ENRICHED_VOCABULARY: # If enriched vocab failed to load, rely on word_categories
        cat_words = [w.lower() for w in word_categories.get(category_name, []) if w.lower() not in exclude_list]
        return random.sample(cat_words, min(count, len(cat_words))) if len(cat_words) >= count else []

    rarity_to_complexity_map = {
        'very_common': (1, 2), 'common': (3, 4), 'somewhat_common': (5, 6),
        'uncommon': (7, 8), 'rare': (9, 9), 'very_rare': (10, 10), 'extremely_rare': (10,10)
    }
    min_c, max_c = rarity_to_complexity_map.get(target_rarity_name, (1, 10)) # Default to wide range

    candidate_words = [word for word, data in ENRICHED_VOCABULARY.items()
                       if data.get("category") == category_name and word not in exclude_list and
                       min_c <= data.get("complexity_score", 5) <= max_c] # Default complexity 5 if not present

    if len(candidate_words) >= count: return random.sample(candidate_words, count)
    else:
        logger.debug(f"Not enough words for cat '{category_name}', rarity '{target_rarity_name}'. Found {len(candidate_words)}, need {count}. Widening rarity search for this category.")
        # Fallback: try to get any words from the category, ignoring rarity, if specific rarity fails
        available_fallback = [word for word, data in ENRICHED_VOCABULARY.items()
                              if data.get("category") == category_name and word not in exclude_list]
        if len(available_fallback) >= count:
            return random.sample(available_fallback, count)
        elif available_fallback: # Return what we have if less than count
             return random.sample(available_fallback, len(available_fallback))
        else: # Truly no words in this category or all excluded
             return []


def _generate_groups_by_category(num_groups_to_gen: int, words_per_group_val: int, used_words_global: Set[str], generation_params: Dict[str, Any]) -> Optional[Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]]:
    target_word_rarity = generation_params.get('word_rarity', 'common')
    all_cat_names = list(word_categories.keys()) # Use the populated word_categories
    if not all_cat_names:
        logger.error("CatGen: No categories available in word_categories. Cannot generate groups.")
        return None
    if len(all_cat_names) < num_groups_to_gen:
        logger.warning(f"CatGen: Not enough distinct categories ({len(all_cat_names)}) to generate {num_groups_to_gen} groups. Will use categories with replacement if necessary, or fail if only one category.")
        if num_groups_to_gen > 1 and len(all_cat_names) == 1: # Cannot make distinct groups
            selected_cat_names = [all_cat_names[0]] * num_groups_to_gen # Use the same category
        elif len(all_cat_names) == 0:
            return None # Should be caught by the check above, but defensive
        else: # Sample with replacement if needed, or just use available if less
            selected_cat_names = random.choices(all_cat_names, k=num_groups_to_gen) if len(all_cat_names) < num_groups_to_gen else random.sample(all_cat_names, num_groups_to_gen)
    else:
        selected_cat_names = random.sample(all_cat_names, num_groups_to_gen)

    solution_groups, all_words_list, descriptions, diff_indices = {}, [], {}, {}
    current_puzzle_used_words = set(used_words_global) # Make a copy to modify locally

    for i, cat_name in enumerate(selected_cat_names):
        group_id = f"group_{i+1}"
        # Get words, ensuring no reuse from *within* this puzzle generation attempt
        group_words = _get_words_by_target_rarity(cat_name, target_word_rarity, words_per_group_val, current_puzzle_used_words)

        if len(group_words) < words_per_group_val:
            logger.warning(f"CatGen: Failed to get {words_per_group_val} words for category '{cat_name}', rarity '{target_word_rarity}'. Found {len(group_words)} words. Aborting this group generation.")
            # Attempt to find a different category if this one failed and we have others
            # This part can be complex; for now, we'll let it fail the puzzle gen if one group fails
            return None # Signal failure for the entire puzzle generation if one group can't be formed

        solution_groups[group_id] = sorted(group_words)
        all_words_list.extend(group_words)
        current_puzzle_used_words.update(group_words) # Add newly selected words to used set for this puzzle
        descriptions[group_id] = connection_descriptions.get('same_category', "Category") + f": {cat_name.replace('_', ' ').capitalize()}"
        diff_indices[group_id] = i

    if len(solution_groups) < num_groups_to_gen:
        logger.warning(f"CatGen: Could not form all {num_groups_to_gen} groups successfully.")
        return None
    return solution_groups, all_words_list, descriptions, diff_indices

# --- Placeholder functions for other connection types ---
# !!! YOU NEED TO IMPLEMENT THE LOGIC FOR THESE FUNCTIONS !!!
def _generate_groups_by_starting_letter(num_groups: int, words_per_group: int, used_words: Set[str], params: Dict[str, Any]) -> Optional[Tuple[Dict, List, Dict, Dict]]:
    logger.warning(f"Word selection logic for 'begins_with' (starting letter) is a STUB. Using fallback.")
    return None # This will trigger fallback

def _generate_groups_by_ending_letter(num_groups: int, words_per_group: int, used_words: Set[str], params: Dict[str, Any]) -> Optional[Tuple[Dict, List, Dict, Dict]]:
    logger.warning(f"Word selection logic for 'ends_with' is a STUB. Using fallback.")
    return None

def _generate_groups_by_rhyme(num_groups: int, words_per_group: int, used_words: Set[str], params: Dict[str, Any]) -> Optional[Tuple[Dict, List, Dict, Dict]]:
    logger.warning(f"Word selection logic for 'rhyming_words' is a STUB. Using fallback.")
    return None

def _generate_groups_by_anagram(num_groups: int, words_per_group: int, used_words: Set[str], params: Dict[str, Any]) -> Optional[Tuple[Dict, List, Dict, Dict]]:
    logger.warning(f"Word selection logic for 'anagrams' is a STUB. Using fallback.")
    return None
# --- Add more placeholder stubs for other connection types as needed ---

def _generate_fallback_groups(num_words_needed_val: int, original_connection_type: str, puzzle_params_for_fallback: Dict[str, Any]) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]:
    logger.warning(f"[FALLBACK-GROUPS] Using 'same_category' logic as fallback for original request '{original_connection_type}'.")
    fallback_params = {**puzzle_params_for_fallback, 'word_rarity': puzzle_params_for_fallback.get('word_rarity', 'common')}
    # Pass an empty set for used_words initially for fallback category generation
    category_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set(), fallback_params)
    if category_data: return category_data
    else:
        logger.critical("CRITICAL FALLBACK: _generate_groups_by_category FAILED. Using dummy.")
        dummy_solution = {f"g{i+1}": [f"err_w{i*WORDS_PER_GROUP+j+1}" for j in range(WORDS_PER_GROUP)] for i in range(NUM_GROUPS)}
        dummy_words = [w for grp in dummy_solution.values() for w in grp]
        return dummy_solution, dummy_words, {k:"Error Group" for k in dummy_solution}, {k:i for i,k in enumerate(dummy_solution)}

def get_fallback_prediction(params: Dict[str, Any], actual_selected_words: Optional[List[str]] = None) -> Dict[str, Any]:
    complexity = connection_types.get(params.get('connection_type', 'same_category'), 1)
    rarity_score_to_use = word_rarity_levels.get(params.get('word_rarity', 'common'), 2)
    if actual_selected_words and ENRICHED_VOCABULARY: # Check if ENRICHED_VOCABULARY is populated
        complexities = [ENRICHED_VOCABULARY.get(w.lower(), {}).get('complexity_score', 5) for w in actual_selected_words]
        if complexities: # Ensure complexities list is not empty
            avg_word_complexity_score = sum(complexities) / len(complexities)
            if avg_word_complexity_score <= 1.5: rarity_score_to_use = 1
            elif avg_word_complexity_score <= 3: rarity_score_to_use = 2
            elif avg_word_complexity_score <= 5: rarity_score_to_use = 4
            elif avg_word_complexity_score <= 7: rarity_score_to_use = 6
            elif avg_word_complexity_score <= 9: rarity_score_to_use = 8
            else: rarity_score_to_use = 10
    score = complexity + rarity_score_to_use
    est_time = 20 + score * 4 + (params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP) - (NUM_GROUPS * WORDS_PER_GROUP)) * 0.5 # Ensure consistent num_words
    difficulty = "medium"
    if est_time <= EASY_THRESHOLD / 2: difficulty = "easy"
    elif est_time <= MEDIUM_THRESHOLD /1.5 : difficulty = "medium"
    else: difficulty = "hard"
    logger.warning(f"[FALLBACK-PRED] Rule-based: Score={score}, EstTime={est_time:.1f}s, Diff='{difficulty}' for params: {params}")
    return {'predicted_solve_time': round(est_time, 2), 'difficulty': difficulty, 'is_fallback': True}

def predict_difficulty_for_params(params: Dict[str, Any], actual_selected_words: Optional[List[str]] = None) -> Dict[str, Any]:
    if not model_pipeline or not ALL_EXPECTED_INPUT_FEATURES:
        return get_fallback_prediction(params, actual_selected_words)
    try:
        feature_data_dict = {name: 0.0 for name in ALL_EXPECTED_INPUT_FEATURES} # Initialize all expected features
        word_rarity_for_model_input = word_rarity_levels.get(params.get('word_rarity'), 5) # Default if not found
        if actual_selected_words and ENRICHED_VOCABULARY:
            complexities = [ENRICHED_VOCABULARY.get(w.lower(), {}).get('complexity_score', 5) for w in actual_selected_words]
            if complexities:
                avg_word_complexity_score = sum(complexities) / len(complexities)
                if avg_word_complexity_score <= 1.5: word_rarity_for_model_input = 1
                elif avg_word_complexity_score <= 3: word_rarity_for_model_input = 2
                elif avg_word_complexity_score <= 5: word_rarity_for_model_input = 4
                elif avg_word_complexity_score <= 7: word_rarity_for_model_input = 6
                elif avg_word_complexity_score <= 9: word_rarity_for_model_input = 8
                else: word_rarity_for_model_input = 10

        # Ensure all necessary parameters for the model are present or have defaults
        feature_data_dict.update({
            'num_words': float(params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP)),
            'connection_complexity': float(connection_types.get(params.get('connection_type'), 5)),
            'word_rarity_value': float(word_rarity_for_model_input),
            'semantic_distance': float(params.get('semantic_distance', 5.0)), # Default semantic distance
            'time_of_day': float(datetime.datetime.now().hour), # Example, might not be a strong feature
            # Add defaults for any other features your model expects if not in params
            'hints_used': 0.0, # Assuming for new puzzle generation
            'num_players': 50.0, # Example default
            'completions': 40.0, # Example default
            'completion_rate': 0.80, # Example default
            'attempt_count': 2.0, # Example default
            'time_before_first_attempt': 10.0, # Example default
            'hover_count': float(params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP) * 1.5), # Example default
            'abandonment_rate': 0.20, # Example default
            'competitiveness_score': 5.0, # Example default
            'frustration_score': 3.0, # Example default
            'satisfaction_score': 7.0, # Example default
            'learning_value': 5.0, # Example default
            'engagement_score': 6.0, # Example default
            'replayability_score': 4.0, # Example default
            'avg_attempts_before_success': 1.5 # Example default
        })

        # Create the feature vector in the order expected by the model
        final_feature_values = [feature_data_dict.get(name, 0.0) for name in ALL_EXPECTED_INPUT_FEATURES]
        if len(final_feature_values) != len(ALL_EXPECTED_INPUT_FEATURES):
            logger.error(f"Feature length mismatch for ML. Expected {len(ALL_EXPECTED_INPUT_FEATURES)}, got {len(final_feature_values)}. ALL_EXPECTED_INPUT_FEATURES: {ALL_EXPECTED_INPUT_FEATURES}, feature_data_dict keys: {list(feature_data_dict.keys())}")
            return get_fallback_prediction(params, actual_selected_words)

        predict_df = pd.DataFrame([final_feature_values], columns=ALL_EXPECTED_INPUT_FEATURES)
        predicted_time = float(model_pipeline.predict(predict_df)[0])

        difficulty = "hard" # Default to hard
        if predicted_time < EASY_THRESHOLD: difficulty = "easy"
        elif predicted_time < MEDIUM_THRESHOLD: difficulty = "medium"

        logger.info(f"ML Prediction: Time={predicted_time:.2f}s, Diff='{difficulty}' for Params: {params} (RarityVal for Model: {word_rarity_for_model_input})")
        return {'predicted_solve_time': round(predicted_time, 2), 'difficulty': difficulty, 'is_fallback': False}
    except Exception as e:
        logger.error(f"Error in ML predict_difficulty_for_params: {e}", exc_info=True)
        return get_fallback_prediction(params, actual_selected_words)

def generate_solvable_puzzle(target_difficulty: str, user_performance_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    puzzle_id = str(uuid.uuid4())
    num_words_total = NUM_GROUPS * WORDS_PER_GROUP
    logger.info(f"\n--- Generating Puzzle. Target Diff: '{target_difficulty.upper()}' ---")
    if user_performance_summary: logger.info(f"User Perf for '{target_difficulty}': {user_performance_summary}")
    else: logger.info("No user performance summary for this generation.")

    generation_params, predicted_result_for_params, found_matching_params = {}, {}, False
    best_params_so_far, best_prediction_so_far = None, None
    param_candidates = []
    actual_search_difficulty = target_difficulty # Start with the requested difficulty

    # Adjust search difficulty based on user performance (nudging)
    if user_performance_summary and user_performance_summary.get("plays", 0) >= 2: # Only nudge if enough data
        win_rate = user_performance_summary.get("win_rate", 0.5); avg_hints = user_performance_summary.get("avg_hints", MAX_HINTS / 2.0)
        if target_difficulty=="easy" and win_rate>0.9 and avg_hints<0.2: actual_search_difficulty="medium"; logger.info("Nudge: Easy->Medium based on user performance.")
        elif target_difficulty=="medium":
            if win_rate>0.75 and avg_hints<0.5: actual_search_difficulty="hard"; logger.info("Nudge: Medium->Hard based on user performance.")
            elif win_rate<0.3 and avg_hints>(MAX_HINTS*0.66): actual_search_difficulty="easy"; logger.info("Nudge: Medium->Easy based on user performance.")
        elif target_difficulty=="hard" and win_rate<0.25 and avg_hints>(MAX_HINTS*0.66): actual_search_difficulty="medium"; logger.info("Nudge: Hard->Medium based on user performance.")

    # Populate parameter candidates based on the (potentially nudged) actual_search_difficulty
    if actual_search_difficulty == 'easy':
        param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd}
            for ct in ['same_category', 'begins_with', 'ends_with', 'contains_substring'] # Easier connection types
            for wr in ['very_common', 'common'] # Easier word rarities
            for sd in [random.uniform(1,2), random.uniform(2,3.5)]] * 5)
    elif actual_search_difficulty == 'medium':
        param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd}
            for ct in ['same_category', 'syllable_count', 'rhyming_words', 'letter_pattern', 'homophones']
            for wr in ['common', 'somewhat_common', 'uncommon']
            for sd in [random.uniform(2.5,4.5), random.uniform(4,6.5)]] * 4)
    else: # hard or if actual_search_difficulty is not 'easy'/'medium'
        param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd}
            for ct in ['conceptual_relation', 'synonym_groups', 'antonym_groups', 'anagrams', 'metaphorical_relation', 'multiple_rules'] # Harder types
            for wr in ['uncommon', 'rare', 'very_rare'] # Harder rarities
            for sd in [random.uniform(5,7), random.uniform(6.5,9)]] * 3)

    if not param_candidates: # Should not happen with the above logic, but as a safeguard
        param_candidates = [{'connection_type': 'same_category', 'word_rarity': 'common', 'semantic_distance': 3.0}] * MAX_GENERATION_ATTEMPTS
    random.shuffle(param_candidates)

    for attempt in range(min(MAX_GENERATION_ATTEMPTS, len(param_candidates))):
        current_params = param_candidates[attempt].copy() # Use a copy to avoid modifying the list items directly
        current_params['num_words'] = num_words_total # Ensure num_words is set
        if current_params.get('connection_type') not in connection_types or current_params.get('word_rarity') not in word_rarity_levels:
            logger.debug(f"Skipping invalid param candidate: {current_params}")
            continue

        current_prediction = predict_difficulty_for_params(current_params)

        # Update best_params_so_far if current is better (prefers non-fallback, then exact match)
        if not best_prediction_so_far or \
           (not current_prediction.get('is_fallback', True) and best_prediction_so_far.get('is_fallback', True)) or \
           (current_prediction.get('difficulty') == target_difficulty and not current_prediction.get('is_fallback', True) and
            (best_prediction_so_far.get('is_fallback', True) or best_prediction_so_far.get('difficulty') != target_difficulty)):
            best_params_so_far = current_params
            best_prediction_so_far = current_prediction

        if current_prediction.get('difficulty') == target_difficulty and not current_prediction.get('is_fallback', True):
            generation_params = current_params
            predicted_result_for_params = current_prediction
            found_matching_params = True
            logger.info(f"--> Target '{target_difficulty}' MET by ML on attempt {attempt+1}. Params: {generation_params}")
            break # Found an exact ML match

    if not found_matching_params:
        logger.warning(f"No exact ML match for '{target_difficulty}'. Using best found: {best_params_so_far} (Pred: {best_prediction_so_far})")
        if best_params_so_far and best_prediction_so_far:
            generation_params = best_params_so_far
            predicted_result_for_params = best_prediction_so_far
        else: # Should only happen if all param_candidates were invalid or ML always failed catastrophically
            logger.error("No valid ML predictions or candidates. Using hardcoded fallback parameters for generation.")
            generation_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total, 'semantic_distance': 3.0}
            predicted_result_for_params = get_fallback_prediction(generation_params) # Get a prediction for these defaults

    final_connection_type = generation_params.get('connection_type', 'same_category')
    logger.info(f"--- Finalizing words. Connection: '{final_connection_type}', TargetRarity: '{generation_params.get('word_rarity')}' ---")

    word_selection_data = None
    if final_connection_type == 'same_category':
        word_selection_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'begins_with':
        word_selection_data = _generate_groups_by_starting_letter(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'ends_with':
        word_selection_data = _generate_groups_by_ending_letter(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'rhyming_words':
        word_selection_data = _generate_groups_by_rhyme(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'anagrams':
        word_selection_data = _generate_groups_by_anagram(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    else: logger.warning(f"Word selection logic for '{final_connection_type}' is not specifically implemented. Using fallback if general attempt fails.")

    if not word_selection_data:
        logger.info(f"Primary word selection failed or not implemented for '{final_connection_type}'. Using fallback group generation.")
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
            _generate_fallback_groups(num_words_total, final_connection_type, generation_params)
    else:
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = word_selection_data

    if not all_words_for_grid or len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
        logger.critical(f"CRITICAL: Puzzle assembly error. Words: {len(all_words_for_grid if all_words_for_grid else [])}/{num_words_total}, Groups: {len(solution_groups)}/{NUM_GROUPS}. Emergency Fallback to ensure puzzle integrity.")
        try:
            # Use very basic params for emergency, ignore original generation_params for this
            emergency_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total}
            solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
                _generate_fallback_groups(num_words_total, "same_category", emergency_params)
            if not all_words_for_grid or len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
                 raise ValueError("Emergency fallback also failed to produce a valid puzzle structure.")
        except Exception as e_fall:
            logger.critical(f"Emergency fallback group generation also failed: {e_fall}", exc_info=True)
            # As a last resort, create an absolutely minimal dummy puzzle to prevent crashes
            solution_groups = {f"dummy_group_{i+1}": [f"word{i*WORDS_PER_GROUP+j+1}" for j in range(WORDS_PER_GROUP)] for i in range(NUM_GROUPS)}
            all_words_for_grid = [word for group in solution_groups.values() for word in group]
            group_descriptions = {k: "Dummy Group" for k in solution_groups}
            difficulty_index_map_for_puzzle = {k: i for i, k in enumerate(solution_groups)}
            generation_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total} # Reset gen_params for dummy
            predicted_result_for_params = get_fallback_prediction(generation_params, all_words_for_grid) # Get prediction for dummy

    # Re-predict difficulty with the actual words selected for the grid if not a dummy last resort
    if not (solution_groups and solution_groups.get("dummy_group_1")): # if not the absolute last resort dummy
        final_puzzle_difficulty_data = predict_difficulty_for_params(generation_params, actual_selected_words=all_words_for_grid)
    else: # It is the last resort dummy
        final_puzzle_difficulty_data = predicted_result_for_params # use prediction from dummy creation

    random.shuffle(all_words_for_grid)
    active_puzzles[puzzle_id] = {
        "puzzle_id": puzzle_id, "words_on_grid": [w.upper() for w in all_words_for_grid],
        "solution": {k: [w.upper() for w in v] for k,v in solution_groups.items()},
        "descriptions": group_descriptions,
        "difficulty": final_puzzle_difficulty_data.get('difficulty', target_difficulty), # Use target_difficulty as ultimate fallback
        "predicted_solve_time": final_puzzle_difficulty_data.get('predicted_solve_time', -1.0),
        "is_fallback_prediction": final_puzzle_difficulty_data.get('is_fallback', True),
        "parameters": {**generation_params, "difficulty_index_map": difficulty_index_map_for_puzzle},
        "creation_time": datetime.datetime.now(), "is_daily": False
    }
    logger.info(f"--- Puzzle {puzzle_id} (Target: {target_difficulty}, Final Gen Diff: {active_puzzles[puzzle_id]['difficulty']}) created. ---")
    return {"puzzle_id": puzzle_id, "words": active_puzzles[puzzle_id]["words_on_grid"], "difficulty": active_puzzles[puzzle_id]["difficulty"]}

def get_or_generate_daily_challenge() -> dict:
    cleanup_old_puzzles(); today_str = datetime.date.today().isoformat(); daily_puzzle_id = f"daily_{today_str}"
    if daily_puzzle_id not in active_puzzles:
        logger.info(f"Generating new daily for {today_str}"); original_random_state = random.getstate(); random.seed(today_str)
        # Daily challenges could have their own specific generation logic or target difficulty
        temp_data = generate_solvable_puzzle(target_difficulty="medium") # Example: daily is always medium
        temp_id = temp_data["puzzle_id"]
        if temp_id in active_puzzles: # Ensure the generated puzzle was stored
            daily_details = active_puzzles.pop(temp_id) # Remove temp, re-key as daily
            daily_details.update({"puzzle_id": daily_puzzle_id, "is_daily": True, "difficulty": "Daily Challenge"}) # Standardize daily difficulty display
            active_puzzles[daily_puzzle_id] = daily_details
        else: # Should not happen if generate_solvable_puzzle works
            random.setstate(original_random_state); # Restore random state before raising
            raise ValueError("Daily challenge temporary puzzle generation failed to store.")
        random.setstate(original_random_state) # Restore random state
    else: logger.info(f"Returning existing daily challenge for {today_str}")
    puzzle = active_puzzles[daily_puzzle_id]
    return {"puzzle_id": puzzle["puzzle_id"], "words": puzzle["words_on_grid"],
            "difficulty": puzzle["difficulty"], "is_daily": True }

def check_puzzle_answer(puzzle_id: str, user_groups_attempt: Dict[str, List[str]]) -> Dict[str, Any]:
    if puzzle_id not in active_puzzles: return {"correct": False, "message": "Invalid or expired puzzle.", "solved_groups": {}}
    data = active_puzzles[puzzle_id]; solution = data["solution"]; descriptions = data["descriptions"]
    diff_idx_map = data.get("parameters", {}).get("difficulty_index_map", {})
    server_solved_groups = data.setdefault("_server_solved_group_keys", set()) # Get or initialize
    attempt_key = next(iter(user_groups_attempt), None) # Get the first key from the attempt dict
    if not attempt_key: return {"correct": False, "message": "No attempt data provided.", "solved_groups": {}}
    attempt_words = sorted([w.upper() for w in user_groups_attempt[attempt_key]])
    if len(attempt_words) != WORDS_PER_GROUP: return {"correct": False, "message": f"Please select exactly {WORDS_PER_GROUP} words.", "solved_groups": {}}

    for group_key_solution, correct_words_solution in solution.items():
        if attempt_words == correct_words_solution: # Assumes solution words are already sorted and uppercased
            if group_key_solution in server_solved_groups:
                return {"correct": False, "message": f"This group ('{descriptions.get(group_key_solution, 'Group')}') has already been solved!", "solved_groups": {}} # Provide more context
            server_solved_groups.add(group_key_solution)
            # active_puzzles[puzzle_id]["_server_solved_group_keys"] = server_solved_groups # Not strictly needed as setdefault modifies in place
            return {"correct": True, "message": f"Correct! You found: {descriptions.get(group_key_solution, 'a group')}",
                    "solved_groups": {group_key_solution: {"description": descriptions.get(group_key_solution, "Unknown Category"),
                                               "difficulty_index": diff_idx_map.get(group_key_solution,0)}}} # Default diff_index
    return {"correct": False, "message": "That's not one of the groups. Try again!", "solved_groups": {}}

def get_puzzle_hint(puzzle_id: str, solved_group_keys_from_client: Optional[List[str]] = None) -> Dict[str, Any]:
    if solved_group_keys_from_client is None: solved_group_keys_from_client = []
    if puzzle_id not in active_puzzles: return {"hint": None, "message": "Invalid or expired puzzle.", "words": []}

    puzzle_data = active_puzzles[puzzle_id]
    actual_solution = puzzle_data.get("solution", {})
    actual_descriptions = puzzle_data.get("descriptions", {})
    all_words_on_grid_upper = set(puzzle_data.get("words_on_grid", [])) # Assumes words_on_grid are already upper

    # Combine server-known solved groups with any client might have solved but not yet synced (defensive)
    server_solved_groups = puzzle_data.get("_server_solved_group_keys", set())
    combined_solved_keys = server_solved_groups.union(set(solved_group_keys_from_client))

    unsolved_groups_details = {
        k: {"words_upper": v, "words_lower": [w.lower() for w in v], "description": actual_descriptions.get(k, "A Group")}
        for k, v in actual_solution.items() if k not in combined_solved_keys
    }

    if not unsolved_groups_details: return {"hint": None, "message": "Congratulations, all groups have been solved!", "words": []}

    target_group_key = random.choice(list(unsolved_groups_details.keys()))
    target_group_info = unsolved_groups_details[target_group_key]
    hint_text = f"Hint for the group: '{target_group_info['description']}'." # Initial hint text
    words_to_highlight_on_grid = []

    if w2v_model and GENSIM_AVAILABLE:
        try:
            # Try to find 1 or 2 words from the target group that are in the w2v model's vocab
            words_from_group_in_vocab = [w for w in target_group_info["words_lower"] if w in w2v_model]
            if words_from_group_in_vocab:
                anchor_words_for_w2v = random.sample(words_from_group_in_vocab, k=min(len(words_from_group_in_vocab), 2)) # Use up to 2 words
                similar_candidates = w2v_model.most_similar(positive=anchor_words_for_w2v, topn=20)
                found_external_hint_word = None
                for sim_word_lower, _ in similar_candidates:
                    sim_word_upper = sim_word_lower.upper()
                    # Ensure hint word is not one of the puzzle words or part of any solution group
                    is_puzzle_word = sim_word_upper in all_words_on_grid_upper
                    is_in_any_solution_group = any(sim_word_upper in sol_group for sol_group in actual_solution.values())

                    if sim_word_lower.isalpha() and len(sim_word_lower) > 2 and not is_puzzle_word and not is_in_any_solution_group:
                        found_external_hint_word = sim_word_lower
                        break
                if found_external_hint_word:
                    hint_text = f"The group '{target_group_info['description']}' might relate to concepts like '{found_external_hint_word.capitalize()}' (this word is not in the puzzle)."
                    # As a bonus, highlight one of the anchor words from the group on the grid
                    words_to_highlight_on_grid.append(random.choice(anchor_words_for_w2v).upper())
                else:
                    logger.info("W2V: No suitable distinct external hint word found for group '%s'. Falling back to revealing words from group.", target_group_key)
            else:
                logger.info("W2V: None of the words from group '%s' are in the Word2Vec vocabulary.", target_group_key)
        except Exception as e:
            logger.error(f"Error during Word2Vec hint generation for group '%s': {e}", target_group_key, exc_info=True)
    
    # If W2V hint wasn't generated or failed, use basic reveal hint
    if not words_to_highlight_on_grid: # Check if W2V already decided to highlight something
        logger.info("Using basic hint (revealing words from group '%s').", target_group_key)
        # Reveal 1 or 2 words from the target group
        num_to_reveal = random.randint(1, min(len(target_group_info["words_upper"]), 2)) # Reveal 1 or 2
        words_to_highlight_on_grid = random.sample(target_group_info["words_upper"], num_to_reveal)
        if words_to_highlight_on_grid:
            hint_text = f"Hint for '{target_group_info['description']}': This group includes {', '.join(words_to_highlight_on_grid)}."
        else: # Should not happen if group has words
            hint_text = f"Try to find words that fit the category: '{target_group_info['description']}'."

    return {"hint": hint_text, "words": words_to_highlight_on_grid, "message": "Hint provided."}

def cleanup_old_puzzles():
    current_time_dt = datetime.datetime.now(); today_date = datetime.date.today()
    puzzles_to_delete = []
    for pid, data in list(active_puzzles.items()): # Iterate over a copy of items for safe deletion
        creation_time_dt = data.get("creation_time")
        # Ensure creation_time_dt is a datetime object
        if not isinstance(creation_time_dt, datetime.datetime):
            if isinstance(creation_time_dt, (float, int)): # If it's a timestamp
                try: creation_time_dt = datetime.datetime.fromtimestamp(creation_time_dt)
                except (OSError, TypeError, ValueError) as ts_err: # Handle potential errors with fromtimestamp
                    logger.warning(f"Puzzle {pid} has invalid timestamp {data.get('creation_time')}. Error: {ts_err}. Skipping cleanup for this puzzle."); continue
            else: # If it's some other non-datetime type
                logger.warning(f"Puzzle {pid} has invalid creation_time type {type(creation_time_dt)}. Skipping cleanup for this puzzle."); continue

        if pid.startswith("daily_"):
            try:
                puzzle_date_str = pid.replace("daily_", "")
                if datetime.date.fromisoformat(puzzle_date_str) < today_date:
                    puzzles_to_delete.append(pid)
            except ValueError: logger.error(f"Malformed daily puzzle ID for cleanup: {pid}. Could not parse date.")
        elif (current_time_dt - creation_time_dt).total_seconds() > MAX_PUZZLE_AGE_SECONDS:
            puzzles_to_delete.append(pid)

    for pid in puzzles_to_delete:
        if pid in active_puzzles:
            del active_puzzles[pid]
            logger.info(f"Cleaned up old puzzle: {pid}")

if __name__ == "__main__":
    logger.info("Testing puzzle_logic.py standalone functions...")
    # Ensure models are loaded for testing if possible, or expect fallbacks
    if not model_pipeline: logger.warning("ML Model not loaded for standalone test, expect fallback predictions.")
    if not w2v_model: logger.warning("Word2Vec Model not loaded for standalone test, expect fallback hints.")

    mock_good_perf_medium = {"plays": 5, "win_rate": 0.9, "avg_hints": 0.1}
    mock_struggling_perf_medium = {"plays": 5, "win_rate": 0.2, "avg_hints": 2.5}

    for diff_test in ["easy", "medium", "hard"]:
        try:
            print(f"\n--- Testing Regular Puzzle Generation ({diff_test.capitalize()}) ---")
            # Pass performance summary only for medium to test nudging logic specifically
            perf_summary_for_test = mock_good_perf_medium if diff_test == "medium" else (mock_struggling_perf_medium if diff_test == "hard" else None)

            puzzle = generate_solvable_puzzle(target_difficulty=diff_test, user_performance_summary=perf_summary_for_test)
            print(f"{diff_test.capitalize()} Puzzle Client Data: {puzzle}")
            if puzzle and puzzle.get('puzzle_id') in active_puzzles:
                server_data = active_puzzles[puzzle['puzzle_id']]
                print(f"  Server Data: SolKeys={list(server_data['solution'].keys())}, ActualDiff={server_data['difficulty']}, PredTime={server_data['predicted_solve_time']:.1f}s, IsFallbackPred={server_data['is_fallback_prediction']}")
                print(f"  Generation Params: {server_data['parameters']}")

                # Test hint generation for this puzzle
                if server_data['solution']:
                    hint_result = get_puzzle_hint(puzzle['puzzle_id'])
                    print(f"  Hint Test: '{hint_result.get('hint')}', Highlight: {hint_result.get('words')}")
            else:
                print(f"  Failed to generate or store {diff_test} puzzle.")
        except Exception as e:
            print(f"Error generating {diff_test} puzzle: {e}")
            logger.error(f"Exception during {diff_test} puzzle generation test:", exc_info=True)

    try:
        print("\n--- Testing Daily Challenge ---")
        daily_data = get_or_generate_daily_challenge()
        print(f"Daily Client Data: {daily_data}")
        if daily_data and daily_data.get('puzzle_id') in active_puzzles:
             print(f"  Daily Server Detail: ActualDiff={active_puzzles[daily_data['puzzle_id']]['difficulty']}")
    except Exception as e:
        print(f"Error with daily challenge: {e}")
        logger.error("Exception during daily challenge test:", exc_info=True)

    logger.info("puzzle_logic.py tests completed.")