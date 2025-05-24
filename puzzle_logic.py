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
BEST_MODEL_NAME_FROM_TRAINING = "ridge_regression" # !!! UPDATE THIS !!!
DIFFICULTY_MODEL_FILENAME = os.path.join(MODEL_DIR, f"wordlinks_{BEST_MODEL_NAME_FROM_TRAINING}_model.pkl")
FEATURE_LIST_FILENAME = os.path.join(MODEL_DIR, "feature_list.pkl")

WORD2VEC_ACTUAL_FILENAME = "glove.6B.100d.kv" # !!! UPDATE THIS (e.g., "GoogleNews-vectors-negative300.bin") !!!
WORD2VEC_MODEL_PATH = os.path.join(MODEL_DIR, WORD2VEC_ACTUAL_FILENAME)

ENRICHED_VOCAB_PATH = os.path.join(MODEL_DIR, "enriched_word_data.json")

MAX_GENERATION_ATTEMPTS = 30 
WORDS_PER_GROUP = 4
NUM_GROUPS = 4
MAX_PUZZLE_AGE_SECONDS = 3 * 3600
MAX_HINTS = 3

# --- Load Difficulty Prediction ML Model Pipeline ---
model_pipeline = None
ALL_EXPECTED_INPUT_FEATURES = [] # This will be populated from feature_list.pkl
try:
    if not os.path.isdir(MODEL_DIR): logger.warning(f"Model directory '{MODEL_DIR}' does not exist. ML for difficulty disabled.")
    elif not os.path.exists(DIFFICULTY_MODEL_FILENAME): logger.warning(f"Difficulty model '{DIFFICULTY_MODEL_FILENAME}' not found. ML disabled.")
    elif not os.path.exists(FEATURE_LIST_FILENAME): logger.warning(f"Feature list '{FEATURE_LIST_FILENAME}' not found. ML disabled.")
    else:
        model_pipeline = joblib.load(DIFFICULTY_MODEL_FILENAME)
        feature_info = joblib.load(FEATURE_LIST_FILENAME)
        # Ensure correct extraction of feature names based on how they were saved
        numeric_features = feature_info.get('numeric_features', [])
        categorical_features = feature_info.get('categorical_features', [])
        ALL_EXPECTED_INPUT_FEATURES = numeric_features + categorical_features # Order must match training
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
            logger.info(f"Attempting to load Word2Vec model from: {WORD2VEC_MODEL_PATH} (This may take time)...")
            # --- !!! CHOOSE AND UNCOMMENT THE CORRECT LOADING METHOD FOR YOUR W2V FILE !!! ---
            w2v_model = KeyedVectors.load(WORD2VEC_MODEL_PATH) # For .kv files
            # w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True) # For .bin files
            # w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH) # For full .model files (then use .wv)
            logger.info(f"Word2Vec model '{WORD2VEC_MODEL_PATH}' loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Word2Vec model from '{WORD2VEC_MODEL_PATH}': {e}. Word2Vec hints fallback.", exc_info=True)
        w2v_model = None
else:
    logger.warning("Gensim library not installed/found. Word2Vec hints will use basic fallback.")

# --- Load Enriched Vocabulary ---
ENRICHED_VOCABULARY: Dict[str, Dict[str, Any]] = {}
word_categories_hardcoded = { # Fallback
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'brown'],
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'bear', 'wolf'],
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach', 'kiwi', 'watermelon'],
    'kitchen_items': ['knife', 'fork', 'spoon', 'plate', 'bowl', 'cup', 'pan', 'pot', 'oven', 'microwave']
} # Add more defaults for robustness
word_categories: Dict[str, List[str]] = {}

if os.path.exists(ENRICHED_VOCAB_PATH):
    try:
        with open(ENRICHED_VOCAB_PATH, 'r', encoding='utf-8') as f:
            ENRICHED_VOCABULARY = json.load(f)
        logger.info(f"Enriched vocabulary loaded from {ENRICHED_VOCAB_PATH} ({len(ENRICHED_VOCABULARY)} words).")
        
        temp_word_categories = {}
        for word, data in ENRICHED_VOCABULARY.items():
            category = data.get("category", "unknown_in_enriched")
            if category not in temp_word_categories: temp_word_categories[category] = []
            temp_word_categories[category].append(word)
        
        if temp_word_categories and "unknown_in_enriched" not in temp_word_categories:
            word_categories = temp_word_categories
            logger.info(f"Word categories populated from enriched vocabulary ({len(word_categories)} categories).")
        else:
            logger.warning("Enriched vocabulary incomplete; using hardcoded word_categories primarily.")
            word_categories = word_categories_hardcoded
    except Exception as e:
        logger.error(f"Error loading/processing {ENRICHED_VOCAB_PATH}: {e}", exc_info=True)
        word_categories = word_categories_hardcoded
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
    for cat_words_list in word_categories.values(): 
        for word in cat_words_list: all_words.add(word.lower())
    return list(all_words - exclude_words)

def _get_words_by_target_rarity(category_name: str, target_rarity_name: str, count: int, exclude_list: Set[str]) -> List[str]:
    if not ENRICHED_VOCABULARY: 
        cat_words = [w.lower() for w in word_categories.get(category_name, []) if w.lower() not in exclude_list]
        return random.sample(cat_words, min(count, len(cat_words))) if len(cat_words) >= count else []
    rarity_to_complexity_map = {
        'very_common': (1, 2), 'common': (3, 4), 'somewhat_common': (5, 6),
        'uncommon': (7, 8), 'rare': (9, 9), 'very_rare': (10, 10), 'extremely_rare': (10,10)
    }
    min_c, max_c = rarity_to_complexity_map.get(target_rarity_name, (1, 10))
    candidate_words = [word for word, data in ENRICHED_VOCABULARY.items()
                       if data.get("category") == category_name and word not in exclude_list and 
                       min_c <= data.get("complexity_score", 5) <= max_c]
    if len(candidate_words) >= count: return random.sample(candidate_words, count)
    else: 
        logger.debug(f"Not enough words for cat '{category_name}', rarity '{target_rarity_name}'. Found {len(candidate_words)}, need {count}. Widening.")
        available_fallback = [word for word, data in ENRICHED_VOCABULARY.items()
                              if data.get("category") == category_name and word not in exclude_list]
        return random.sample(available_fallback, min(count, len(available_fallback))) if len(available_fallback) >= count else []

def _generate_groups_by_category(num_groups_to_gen: int, words_per_group_val: int, used_words_global: Set[str], generation_params: Dict[str, Any]) -> Optional[Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]]:
    target_word_rarity = generation_params.get('word_rarity', 'common')
    all_cat_names = list(word_categories.keys())
    if len(all_cat_names) < num_groups_to_gen: logger.warning(f"CatGen: Not enough cats ({len(all_cat_names)})"); return None
    selected_cat_names = random.sample(all_cat_names, num_groups_to_gen)
    solution_groups, all_words, descriptions, diff_indices = {}, [], {}, {}
    current_puzzle_used_words = set(used_words_global) 
    for i, cat_name in enumerate(selected_cat_names):
        group_id = f"group_{i+1}"
        group_words = _get_words_by_target_rarity(cat_name, target_word_rarity, words_per_group_val, current_puzzle_used_words)
        if len(group_words) < words_per_group_val:
            logger.warning(f"CatGen: Failed for cat '{cat_name}', rarity '{target_word_rarity}'. Found {len(group_words)} words.")
            return None 
        solution_groups[group_id] = sorted(group_words); all_words.extend(group_words)
        current_puzzle_used_words.update(group_words) 
        descriptions[group_id] = connection_descriptions.get('same_category', "Category") + f": {cat_name.capitalize()}"
        diff_indices[group_id] = i 
    if len(solution_groups) < num_groups_to_gen : return None
    return solution_groups, all_words, descriptions, diff_indices

def _generate_fallback_groups(num_words_needed_val: int, original_connection_type: str, puzzle_params_for_fallback: Dict[str, Any]) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]:
    logger.warning(f"[FALLBACK-GROUPS] Using 'same_category' for type '{original_connection_type}'.")
    fallback_params = {**puzzle_params_for_fallback, 'word_rarity': puzzle_params_for_fallback.get('word_rarity', 'common')}
    category_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set(), fallback_params)
    if category_data: return category_data
    else:
        logger.critical("CRITICAL FALLBACK: _generate_groups_by_category FAILED. Using dummy.")
        dummy_solution = {f"g{i+1}": [f"err_w{i*4+j+1}" for j in range(4)] for i in range(4)}
        dummy_words = [w for grp in dummy_solution.values() for w in grp]
        return dummy_solution, dummy_words, {k:"Error Group" for k in dummy_solution}, {k:i for i,k in enumerate(dummy_solution)}

def get_fallback_prediction(params: Dict[str, Any], actual_selected_words: Optional[List[str]] = None) -> Dict[str, Any]:
    complexity = connection_types.get(params.get('connection_type', 'same_category'), 1)
    rarity_score_to_use = word_rarity_levels.get(params.get('word_rarity', 'common'), 2)
    if actual_selected_words and ENRICHED_VOCABULARY:
        complexities = [ENRICHED_VOCABULARY.get(w.lower(), {}).get('complexity_score', 5) for w in actual_selected_words]
        if complexities:
            avg_word_complexity_score = sum(complexities) / len(complexities)
            if avg_word_complexity_score <= 1.5: rarity_score_to_use = 1 
            elif avg_word_complexity_score <= 3: rarity_score_to_use = 2 
            elif avg_word_complexity_score <= 5: rarity_score_to_use = 4 
            elif avg_word_complexity_score <= 7: rarity_score_to_use = 6 
            elif avg_word_complexity_score <= 9: rarity_score_to_use = 8 
            else: rarity_score_to_use = 10 
    score = complexity + rarity_score_to_use
    est_time = 20 + score * 4 + (params.get('num_words', 16) - 16) * 0.5 
    difficulty = "medium"
    if est_time <= 35: difficulty = "easy"
    elif est_time <= 75: difficulty = "medium"
    else: difficulty = "hard"
    logger.warning(f"[FALLBACK-PRED] Rule-based: Score={score}, EstTime={est_time:.1f}s, Diff='{difficulty}'")
    return {'predicted_solve_time': round(est_time, 2), 'difficulty': difficulty, 'is_fallback': True}

# --- predict_difficulty_for_params (CORRECTED DICTIONARY) ---
def predict_difficulty_for_params(params: Dict[str, Any], actual_selected_words: Optional[List[str]] = None) -> Dict[str, Any]:
    if not model_pipeline or not ALL_EXPECTED_INPUT_FEATURES:
        return get_fallback_prediction(params, actual_selected_words)
    try:
        feature_data_dict = {name: 0.0 for name in ALL_EXPECTED_INPUT_FEATURES}
        
        word_rarity_for_model_input = word_rarity_levels.get(params.get('word_rarity'), 5) 
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
                logger.info(f"ML Input: Avg actual word complexity {avg_word_complexity_score:.2f} -> rarity_value {word_rarity_for_model_input}")
        
        # This is the dictionary that had syntax errors previously
        feature_data_dict.update({
            'num_words': float(params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP)),
            'connection_complexity': float(connection_types.get(params.get('connection_type'), 5)),
            'word_rarity_value': float(word_rarity_for_model_input), # Uses potentially adjusted value
            'semantic_distance': float(params.get('semantic_distance', 5.0)),
            'time_of_day': float(datetime.datetime.now().hour),
            'hints_used': 0.0, # For generation, this is typically 0
            'num_players': 50.0, # Example default, align with your training data
            'completions': 40.0, # Example default
            'completion_rate': 0.80, # Example default (0.0 to 1.0)
            'attempt_count': 2.0, # Example default
            'time_before_first_attempt': 10.0, # Example default
            'hover_count': float(params.get('num_words', 16) * 1.5), # Example default
            'abandonment_rate': 0.20, # Example default (0.0 to 1.0)
            'competitiveness_score': 5.0, # Example default
            'frustration_score': 3.0, # Example default
            'satisfaction_score': 7.0, # Example default
            'learning_value': 5.0, # Example default
            'engagement_score': 6.0, # Example default
            'replayability_score': 4.0, # Example default
            'avg_attempts_before_success': 1.5 # Example default
        })
        # End of the .update() dictionary

        final_feature_values = [feature_data_dict.get(name, 0.0) for name in ALL_EXPECTED_INPUT_FEATURES]
        if len(final_feature_values) != len(ALL_EXPECTED_INPUT_FEATURES):
            logger.error(f"Feature length mismatch. Expected {len(ALL_EXPECTED_INPUT_FEATURES)}, got {len(final_feature_values)}")
            return get_fallback_prediction(params, actual_selected_words)

        predict_df = pd.DataFrame([final_feature_values], columns=ALL_EXPECTED_INPUT_FEATURES)
        predicted_time = float(model_pipeline.predict(predict_df)[0])
        difficulty = "medium"
        if predicted_time < 40: difficulty = "easy"
        elif predicted_time < 80: difficulty = "medium"
        else: difficulty = "hard"
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
    actual_search_difficulty = target_difficulty

    if user_performance_summary and user_performance_summary.get("plays", 0) >= 2: 
        win_rate = user_performance_summary.get("win_rate", 0.5); avg_hints = user_performance_summary.get("avg_hints", MAX_HINTS / 2.0)
        if target_difficulty=="easy" and win_rate>0.9 and avg_hints<0.2: actual_search_difficulty="medium"; logger.info("Nudge: Easy->Medium based on perf.")
        elif target_difficulty=="medium":
            if win_rate>0.75 and avg_hints<0.5: actual_search_difficulty="hard"; logger.info("Nudge: Medium->Hard based on perf.")
            elif win_rate<0.3 and avg_hints>(MAX_HINTS*0.66): actual_search_difficulty="easy"; logger.info("Nudge: Medium->Easy based on perf.")
        elif target_difficulty=="hard" and win_rate<0.25 and avg_hints>(MAX_HINTS*0.66): actual_search_difficulty="medium"; logger.info("Nudge: Hard->Medium based on perf.")
    
    if actual_search_difficulty == 'easy': param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd} for ct in ['same_category', 'begins_with'] for wr in ['very_common', 'common'] for sd in [random.uniform(1,3)]] * 8)
    elif actual_search_difficulty == 'medium': param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd} for ct in ['same_category', 'syllable_count', 'rhyming_words'] for wr in ['common', 'somewhat_common'] for sd in [random.uniform(3,6)]] * 8)
    else: param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd} for ct in ['conceptual_relation', 'synonym_groups', 'anagrams'] for wr in ['uncommon', 'rare'] for sd in [random.uniform(5,9)]] * 8)
    if not param_candidates: param_candidates = [{'connection_type': 'same_category', 'word_rarity': 'common', 'semantic_distance': 3.0}] * MAX_GENERATION_ATTEMPTS
    random.shuffle(param_candidates)

    for attempt in range(min(MAX_GENERATION_ATTEMPTS, len(param_candidates))):
        current_params = param_candidates[attempt]; current_params['num_words'] = num_words_total
        if current_params.get('connection_type') not in connection_types or current_params.get('word_rarity') not in word_rarity_levels: continue
        current_prediction = predict_difficulty_for_params(current_params) # Predict based on TARGET params first
        if not best_prediction_so_far or \
           (not current_prediction.get('is_fallback') and best_prediction_so_far.get('is_fallback', True)) or \
           (current_prediction.get('difficulty') == target_difficulty and not current_prediction.get('is_fallback') and \
            (not best_prediction_so_far.get('is_fallback',True) or best_prediction_so_far.get('difficulty') != target_difficulty ) ) :
            best_params_so_far = current_params.copy(); best_prediction_so_far = current_prediction.copy()
        if current_prediction.get('difficulty') == target_difficulty and not current_prediction.get('is_fallback'):
            generation_params = current_params; predicted_result_for_params = current_prediction; found_matching_params = True
            logger.info(f"--> Target '{target_difficulty}' MET by ML on attempt {attempt+1} with params: {generation_params}")
            break
    
    if not found_matching_params:
        logger.warning(f"No exact ML match for '{target_difficulty}'. Using best: {best_params_so_far} (Pred: {best_prediction_so_far})")
        if best_params_so_far and best_prediction_so_far: generation_params, predicted_result_for_params = best_params_so_far, best_prediction_so_far
        else:
            logger.error("No valid ML preds. Hardcoded fallback for gen."); generation_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total, 'semantic_distance': 3.0}
            predicted_result_for_params = get_fallback_prediction(generation_params)

    final_connection_type = generation_params.get('connection_type', 'same_category')
    logger.info(f"--- Finalizing words. Connection: '{final_connection_type}', TargetRarity: '{generation_params.get('word_rarity')}' ---")
    solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = {}, [], {}, {}
    
    word_selection_data = None
    if final_connection_type == 'same_category':
        word_selection_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    # TODO: Add elif blocks for other connection_types, calling specific _generate_groups_by_X functions
    # Each _generate_groups_by_X should use generation_params (esp. 'word_rarity') and ENRICHED_VOCABULARY
    else: logger.warning(f"Word selection for '{final_connection_type}' not fully implemented. Using fallback.")

    if word_selection_data:
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = word_selection_data
    else: 
        logger.info(f"Specific selection failed/not impl for '{final_connection_type}'. Fallback groups.")
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
            _generate_fallback_groups(num_words_total, final_connection_type, generation_params)

    if len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
        logger.critical(f"CRITICAL: Puzzle assembly error. Words: {len(all_words_for_grid)}/{num_words_total}, Groups: {len(solution_groups)}/{NUM_GROUPS}. Emergency Fallback.")
        try:
            solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
                _generate_fallback_groups(num_words_total, "same_category", {'word_rarity': 'common'}) 
            if len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
                 raise ValueError("Emergency fallback also failed.")
        except Exception as e_fall: logger.critical(f"Emergency fallback error: {e_fall}"); raise ValueError("Malformed puzzle gen.") from e_fall
    
    # Predict difficulty based on *actual selected words* for final assignment
    final_puzzle_difficulty_data = predict_difficulty_for_params(generation_params, actual_selected_words=all_words_for_grid)

    random.shuffle(all_words_for_grid)
    active_puzzles[puzzle_id] = {
        "puzzle_id": puzzle_id, "words_on_grid": [w.upper() for w in all_words_for_grid],
        "solution": {k: [w.upper() for w in v] for k,v in solution_groups.items()}, 
        "descriptions": group_descriptions, 
        "difficulty": final_puzzle_difficulty_data.get('difficulty', target_difficulty), 
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
        temp_data = generate_solvable_puzzle(target_difficulty="medium") 
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
    server_solved_groups = data.setdefault("_server_solved_group_keys", set())

    attempt_key = next(iter(user_groups_attempt), None)
    if not attempt_key: return {"correct": False, "message": "No attempt.", "solved_groups": {}}
    attempt_words = sorted([w.upper() for w in user_groups_attempt[attempt_key]])
    if len(attempt_words) != WORDS_PER_GROUP: return {"correct": False, "message": f"Select {WORDS_PER_GROUP} words.", "solved_groups": {}}

    for gk, correct_words in solution.items():
        if attempt_words == correct_words:
            if gk in server_solved_groups:
                return {"correct": False, "message": f"Already found: {descriptions.get(gk, 'Group')}!", "solved_groups": {}}
            server_solved_groups.add(gk)
            active_puzzles[puzzle_id]["_server_solved_group_keys"] = server_solved_groups
            return {"correct": True, "message": f"Correct! Category: {descriptions.get(gk, 'Found')}",
                    "solved_groups": {gk: {"description": descriptions.get(gk, "Unknown"), "difficulty_index": diff_idx_map.get(gk,0)}}}
    return {"correct": False, "message": "Incorrect group.", "solved_groups": {}}

def get_puzzle_hint(puzzle_id: str, solved_group_keys_from_client: Optional[List[str]] = None) -> Dict[str, Any]:
    if solved_group_keys_from_client is None: solved_group_keys_from_client = []
    if puzzle_id not in active_puzzles:
        return {"hint": None, "message": "Invalid or expired puzzle.", "words": []}
    puzzle_data = active_puzzles[puzzle_id]
    actual_solution = puzzle_data.get("solution", {})
    actual_descriptions = puzzle_data.get("descriptions", {})
    all_words_on_grid_upper = set(puzzle_data.get("words_on_grid", []))
    server_solved_groups = puzzle_data.get("_server_solved_group_keys", set(solved_group_keys_from_client))

    unsolved_groups = {k: {"words_upper": v, "words_lower": [w.lower() for w in v], 
                           "description": actual_descriptions.get(k, "A Group")}
                       for k, v in actual_solution.items() if k not in server_solved_groups}
    if not unsolved_groups: return {"hint": None, "message": "All groups solved!", "words": []}
    
    target_group_key = random.choice(list(unsolved_groups.keys()))
    target_group_info = unsolved_groups[target_group_key]
    hint_text = f"Hint for '{target_group_info['description']}'."
    words_to_highlight = []

    if w2v_model and GENSIM_AVAILABLE:
        try:
            anchor_words_lower = random.sample(target_group_info["words_lower"], k=min(len(target_group_info["words_lower"]), 1))
            vocab_source = w2v_model.wv if hasattr(w2v_model, 'wv') and hasattr(w2v_model.wv, 'key_to_index') else w2v_model
            valid_anchor_words = [w for w in anchor_words_lower if w in vocab_source]
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
                    hint_text = f"This group relates to '{target_group_info['description']}'. Think about concepts around '{valid_anchor_words[0].upper()}' or ideas like '{found_w2v_hint_word_upper}' (not in puzzle)."
                    words_to_highlight.append(valid_anchor_words[0].upper())
                else: logger.info("W2V found no suitable distinct hint word for group '%s'.", target_group_key)
            else: logger.info("Anchor words for W2V hint not in vocab for group '%s'.", target_group_key)
        except Exception as e: logger.error(f"Error in W2V hint gen for group '%s': {e}",target_group_key, exc_info=True)
    
    if not words_to_highlight: 
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
             print(f"  Actual Difficulty of Generated Puzzle: {active_puzzles[puzzle_good_perf['puzzle_id']]['difficulty']}")
    except Exception as e: print(f"Error: {e}", exc_info=True)
    
    print("\n--- Testing Personalized Puzzle Generation (Medium - Struggling Perf) ---")
    try:
        puzzle_struggling_perf = generate_solvable_puzzle(target_difficulty="medium", user_performance_summary=mock_struggling_perf_medium)
        print(f"Struggling Perf Medium Puzzle Client Data: {puzzle_struggling_perf}")
        if puzzle_struggling_perf and puzzle_struggling_perf.get('puzzle_id') in active_puzzles:
             print(f"  Server Data: GenParams={active_puzzles[puzzle_struggling_perf['puzzle_id']]['parameters']}")
             print(f"  Actual Difficulty of Generated Puzzle: {active_puzzles[puzzle_struggling_perf['puzzle_id']]['difficulty']}")
    except Exception as e: print(f"Error: {e}", exc_info=True)

    print("\n--- Testing Daily Challenge ---")
    try:
        daily_data = get_or_generate_daily_challenge()
        print(f"Daily Client Data: {daily_data}")
    except Exception as e: print(f"Error with daily challenge: {e}", exc_info=True)
    logger.info("puzzle_logic.py tests completed.")