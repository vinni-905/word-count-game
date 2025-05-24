import random
import uuid
import joblib
import pandas as pd
import os
import datetime
import time
import logging
import json # For loading enriched_word_data.json
from typing import Dict, Any, Optional, List, Tuple, Set

# Gensim import for Word2Vec
try:
    from gensim.models import Word2Vec, KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    KeyedVectors = None 
    Word2Vec = None    

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

MODEL_DIR = "model"
BEST_MODEL_NAME_FROM_TRAINING = "ridge_regression" 
DIFFICULTY_MODEL_FILENAME = os.path.join(MODEL_DIR, f"wordlinks_{BEST_MODEL_NAME_FROM_TRAINING}_model.pkl")
FEATURE_LIST_FILENAME = os.path.join(MODEL_DIR, "feature_list.pkl")
WORD2VEC_ACTUAL_FILENAME = "glove.6B.100d.kv" 
WORD2VEC_MODEL_PATH = os.path.join(MODEL_DIR, WORD2VEC_ACTUAL_FILENAME)
ENRICHED_VOCAB_PATH = os.path.join(MODEL_DIR, "enriched_word_data.json") # Path to our new data

MAX_GENERATION_ATTEMPTS = 30 
WORDS_PER_GROUP = 4
NUM_GROUPS = 4
MAX_PUZZLE_AGE_SECONDS = 3 * 3600
MAX_HINTS = 3

model_pipeline = None
ALL_EXPECTED_INPUT_FEATURES = []
try:
    if os.path.exists(DIFFICULTY_MODEL_FILENAME) and os.path.exists(FEATURE_LIST_FILENAME):
        model_pipeline = joblib.load(DIFFICULTY_MODEL_FILENAME)
        feature_info = joblib.load(FEATURE_LIST_FILENAME)
        ALL_EXPECTED_INPUT_FEATURES = feature_info.get('numeric_features', []) + feature_info.get('categorical_features', [])
        if not ALL_EXPECTED_INPUT_FEATURES: model_pipeline = None; logger.error("Feature list empty. ML for difficulty disabled.")
        else: logger.info(f"Difficulty Prediction ML Model ({DIFFICULTY_MODEL_FILENAME}) loaded.")
    else: logger.warning("Difficulty model or feature list not found. ML for difficulty disabled.")
except Exception as e: logger.error(f"Err loading Difficulty ML: {e}", exc_info=True); model_pipeline = None

w2v_model = None
if GENSIM_AVAILABLE:
    try:
        if os.path.exists(WORD2VEC_MODEL_PATH):
            logger.info(f"Loading W2V model: {WORD2VEC_MODEL_PATH}...")
            w2v_model = KeyedVectors.load(WORD2VEC_MODEL_PATH)
            logger.info(f"W2V model '{WORD2VEC_MODEL_PATH}' loaded.")
        else: logger.warning(f"W2V model '{WORD2VEC_MODEL_PATH}' not found. Hints fallback.")
    except Exception as e: logger.error(f"Err loading W2V: {e}", exc_info=True); w2v_model = None
else: logger.warning("Gensim not installed. W2V hints fallback.")

ENRICHED_VOCABULARY: Dict[str, Dict[str, Any]] = {}
if os.path.exists(ENRICHED_VOCAB_PATH):
    try:
        with open(ENRICHED_VOCAB_PATH, 'r', encoding='utf-8') as f:
            ENRICHED_VOCABULARY = json.load(f)
        logger.info(f"Enriched vocabulary loaded ({len(ENRICHED_VOCABULARY)} words).")
    except Exception as e: logger.error(f"Error loading enriched_word_data.json: {e}", exc_info=True)
else:
    logger.warning(f"'{ENRICHED_VOCAB_PATH}' not found. Word complexity features will be basic.")

# --- Data Definitions ---
# word_categories can now be primarily derived from ENRICHED_VOCABULARY
# but keep a hardcoded version as a fallback if enriched data is missing/empty
word_categories_hardcoded = { # Fallback
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'brown', 'gray', 'teal'],
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'bear', 'wolf', 'fox', 'deer'],
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach', 'kiwi', 'watermelon', 'cherry', 'pear'],
    # Add more for robustness if enriched data fails to load
}
word_categories: Dict[str, List[str]] = {}
if ENRICHED_VOCABULARY:
    for word, data in ENRICHED_VOCABULARY.items():
        category = data.get("category", "unknown")
        if category not in word_categories: word_categories[category] = []
        word_categories[category].append(word) # word is already lowercase from analyzer
    if not word_categories or "unknown" in word_categories: # If enriched data was poor
        logger.warning("Enriched vocabulary was incomplete or had many unknowns; using hardcoded categories.")
        word_categories = word_categories_hardcoded
    else:
        logger.info(f"Word categories populated from enriched vocabulary ({len(word_categories)} categories).")
else:
    logger.warning("Using hardcoded word_categories as enriched vocabulary was not loaded.")
    word_categories = word_categories_hardcoded

connection_types = { 'same_category': 1, 'begins_with': 2, 'ends_with': 3, 'syllable_count': 4, 'synonym_groups': 5, 'antonym_groups': 6, 'compound_words': 7, 'rhyming_words': 7, 'conceptual_relation': 8, 'multiple_rules': 10, 'letter_pattern': 5, 'anagrams': 9, 'homophones': 6, 'contains_substring': 4, 'metaphorical_relation': 9 }
connection_descriptions = { 'same_category': "Same Category", 'begins_with': "Begin With Same Letter", 'ends_with': "End With Same Letter", 'syllable_count': "Same Syllable Count", 'synonym_groups': "Synonyms", 'antonym_groups': "Antonyms", 'compound_words': "Compound Words", 'rhyming_words': "Rhyming Words", 'conceptual_relation': "Conceptual Link", 'multiple_rules': "Multiple Connections", 'letter_pattern': "Shared Letters", 'anagrams': "Anagrams", 'homophones': "Homophones", 'contains_substring': "Contain Substring", 'metaphorical_relation': "Metaphorical Link"}
word_rarity_levels = { 'very_common': 1, 'common': 2, 'somewhat_common': 4, 'uncommon': 6, 'rare': 8, 'very_rare': 10, 'extremely_rare': 12 }
active_puzzles: Dict[str, Dict[str, Any]] = {}

def _get_all_available_words(exclude_words: Optional[Set[str]] = None) -> List[str]:
    # ... (same as before, but now uses the dynamically populated `word_categories`) ...
    if exclude_words is None: exclude_words = set()
    all_words = set()
    for cat_words_list in word_categories.values(): # Uses the potentially rebuilt word_categories
        for word in cat_words_list: all_words.add(word.lower())
    return list(all_words - exclude_words)

def _get_words_by_target_rarity(category_words: List[str], target_rarity_name: str, count: int, exclude_list: Set[str]) -> List[str]:
    """
    Helper to get 'count' words from 'category_words' matching 'target_rarity_name'
    using complexity_score from ENRICHED_VOCABULARY.
    """
    if not ENRICHED_VOCABULARY: # Fallback if no enriched data
        available = [w for w in category_words if w not in exclude_list]
        return random.sample(available, min(count, len(available))) if len(available) >= count else []

    rarity_to_complexity_map = { # Define your mapping
        'very_common': (1, 2), 'common': (3, 4), 'somewhat_common': (5, 6),
        'uncommon': (7, 8), 'rare': (9, 9), 'very_rare': (10, 10), 'extremely_rare': (10,10)
    }
    min_c, max_c = rarity_to_complexity_map.get(target_rarity_name, (1, 10)) # Default to wide range

    candidate_words = []
    for word in category_words:
        if word not in exclude_list:
            word_data = ENRICHED_VOCABULARY.get(word, {})
            complexity_score = word_data.get("complexity_score", 5) # Default if word not in enriched (should not happen if lists are synced)
            if min_c <= complexity_score <= max_c:
                candidate_words.append(word)
    
    if len(candidate_words) >= count:
        return random.sample(candidate_words, count)
    else: # Fallback: not enough words of target rarity, pick from any available in category
        logger.debug(f"Not enough words for rarity '{target_rarity_name}' in provided list. Found {len(candidate_words)}, need {count}. Widening.")
        available_fallback = [w for w in category_words if w not in exclude_list]
        return random.sample(available_fallback, min(count, len(available_fallback))) if len(available_fallback) >= count else []


def _generate_groups_by_category(num_groups_to_gen: int, words_per_group_val: int, used_words_global: Set[str], generation_params: Dict[str, Any]) -> Optional[Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]]:
    target_word_rarity = generation_params.get('word_rarity', 'common') # Get from generation parameters
    
    all_cat_names = list(word_categories.keys())
    if len(all_cat_names) < num_groups_to_gen: logger.warning(f"CatGen: Not enough cats ({len(all_cat_names)}) for {num_groups_to_gen} groups."); return None
        
    selected_cat_names = random.sample(all_cat_names, num_groups_to_gen)
    solution_groups, all_words, descriptions, diff_indices = {}, [], {}, {}
    current_puzzle_used_words = set(used_words_global) 

    for i, cat_name in enumerate(selected_cat_names):
        group_id = f"group_{i+1}"
        words_in_this_category = word_categories.get(cat_name, [])
        
        # Get words matching target rarity, excluding already used for this puzzle
        group_words = _get_words_by_target_rarity(words_in_this_category, target_word_rarity, words_per_group_val, current_puzzle_used_words)
        
        if len(group_words) < words_per_group_val:
            logger.warning(f"CatGen: Failed to get {words_per_group_val} words for cat '{cat_name}' with rarity '{target_word_rarity}'.")
            return None 
            
        solution_groups[group_id] = sorted(group_words)
        all_words.extend(group_words)
        current_puzzle_used_words.update(group_words) 
        descriptions[group_id] = connection_descriptions.get('same_category', "Category") + f": {cat_name.capitalize()}"
        diff_indices[group_id] = i 
    
    if len(solution_groups) < num_groups_to_gen : return None
    return solution_groups, all_words, descriptions, diff_indices


def _generate_fallback_groups(num_words_needed_val: int, original_connection_type: str, puzzle_params_for_fallback: Dict[str, Any]) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]:
    logger.warning(f"[FALLBACK-GROUPS] Using 'same_category' logic as fallback for original request '{original_connection_type}'.")
    # Pass puzzle_params_for_fallback so it can attempt to respect rarity
    category_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set(), puzzle_params_for_fallback) 
    if category_data:
        return category_data
    else: # Ultimate fallback
        logger.critical("CRITICAL FALLBACK: _generate_groups_by_category failed. Using dummy data.")
        # ... (same dummy data generation as before)
        dummy_solution = {f"group_{i+1}": [f"fb_w{i*4+j}" for j in range(4)] for i in range(4)}
        dummy_words = [w for grp in dummy_solution.values() for w in grp]
        dummy_desc = {k: "Fallback Group" for k in dummy_solution}
        dummy_idx = {k: i for i, k in enumerate(dummy_solution)}
        return dummy_solution, dummy_words, dummy_desc, dummy_idx


def get_fallback_prediction(params: Dict[str, Any]) -> Dict[str, Any]:
    # ... (same as your complete version) ...
    logger.warning("[FALLBACK-PRED] Using rule-based difficulty estimation for ML prediction.")
    complexity = connection_types.get(params.get('connection_type', 'same_category'), 1)
    rarity_score = word_rarity_levels.get(params.get('word_rarity'), 2) # Default to 'common' score
    
    # If enriched vocab is available and words were selected, use their avg complexity
    # This path is harder to trigger if params for prediction are set *before* word selection
    if 'selected_puzzle_words_for_ml' in params and ENRICHED_VOCABULARY:
        complexities = [ENRICHED_VOCABULARY.get(w.lower(), {}).get('complexity_score', 5) for w in params['selected_puzzle_words_for_ml']]
        if complexities:
            avg_complexity_of_words = sum(complexities) / len(complexities)
            # Heuristic: map avg_complexity_of_words back to a rarity-like score for the formula
            if avg_complexity_of_words <= 2: rarity_score = 1 # very_common
            elif avg_complexity_of_words <= 4: rarity_score = 2 # common
            elif avg_complexity_of_words <= 6: rarity_score = 4 # somewhat_common
            elif avg_complexity_of_words <= 8: rarity_score = 6 # uncommon
            else: rarity_score = 8 # rare
    else: # Use the target rarity level for estimation
        rarity_score = word_rarity_levels.get(params.get('word_rarity', 'common'), 2)

    score = complexity + rarity_score
    est_time = 20 + score * 4 + (params.get('num_words', 16) - 16) * 0.5
    difficulty = "medium"
    if est_time <= 35: difficulty = "easy"
    elif est_time <= 75: difficulty = "medium"
    else: difficulty = "hard"
    return {'predicted_solve_time': round(est_time, 2), 'difficulty': difficulty, 'is_fallback': True}


def predict_difficulty_for_params(params: Dict[str, Any], actual_selected_words: Optional[List[str]] = None) -> Dict[str, Any]:
    if not model_pipeline or not ALL_EXPECTED_INPUT_FEATURES:
        # Pass actual_selected_words to fallback if available
        if actual_selected_words: params['selected_puzzle_words_for_ml'] = actual_selected_words
        return get_fallback_prediction(params)
    try:
        feature_data_dict = {name: 0.0 for name in ALL_EXPECTED_INPUT_FEATURES}
        
        # Calculate word_rarity_value based on actual_selected_words if provided
        word_rarity_for_model_input = word_rarity_levels.get(params.get('word_rarity'), 5) # Default to target rarity level
        if actual_selected_words and ENRICHED_VOCABULARY:
            complexities = [ENRICHED_VOCABULARY.get(w.lower(), {}).get('complexity_score', 5) for w in actual_selected_words]
            if complexities:
                avg_word_complexity_score = sum(complexities) / len(complexities)
                # Map this avg_word_complexity_score to one of your word_rarity_levels values (1-12)
                # This mapping needs to be defined. For simplicity, we can use a direct scale or bucket it.
                # Example: if complexity score is 1-10, map it to 1-12 range for rarity_value.
                # This is a placeholder, refine this mapping based on your complexity score distribution.
                if avg_word_complexity_score <= 2: word_rarity_for_model_input = 1 # very_common
                elif avg_word_complexity_score <= 4: word_rarity_for_model_input = 2 # common
                elif avg_word_complexity_score <= 6: word_rarity_for_model_input = 4 # somewhat_common
                elif avg_word_complexity_score <= 8: word_rarity_for_model_input = 6 # uncommon
                else: word_rarity_for_model_input = 8 # rare
                logger.info(f"ML Input: Using avg actual word complexity {avg_word_complexity_score:.2f} mapped to rarity value {word_rarity_for_model_input}")


        feature_data_dict.update({
            'num_words': float(params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP)),
            'connection_complexity': float(connection_types.get(params.get('connection_type'), 5)),
            'word_rarity_value': float(word_rarity_for_model_input), # <<< USES CALCULATED/TARGET RARITY
            'semantic_distance': float(params.get('semantic_distance', 5.0)),
            'time_of_day': float(datetime.datetime.now().hour), 'hints_used': 0.0,
            'num_players': 50.0, 'completions': 40.0, 'completion_rate': 0.80, 'attempt_count': 2.0,
            'time_before_first_attempt': 10.0, 'hover_count': float(params.get('num_words', 16) * 1.5),
            'abandonment_rate': 0.20, 'competitiveness_score': 5.0, 'frustration_score': 3.0,
            'satisfaction_score': 7.0, 'learning_value': 5.0, 'engagement_score': 6.0,
            'replayability_score': 4.0, 'avg_attempts_before_success': 1.5
        })
        final_feature_values = [feature_data_dict.get(name, 0.0) for name in ALL_EXPECTED_INPUT_FEATURES]
        if len(final_feature_values) != len(ALL_EXPECTED_INPUT_FEATURES):
            logger.error(f"Feature length mismatch. Expected {len(ALL_EXPECTED_INPUT_FEATURES)}, got {len(final_feature_values)}")
            if actual_selected_words: params['selected_puzzle_words_for_ml'] = actual_selected_words
            return get_fallback_prediction(params)

        predict_df = pd.DataFrame([final_feature_values], columns=ALL_EXPECTED_INPUT_FEATURES)
        predicted_time = float(model_pipeline.predict(predict_df)[0])
        difficulty = "medium"
        if predicted_time < 40: difficulty = "easy"
        elif predicted_time < 80: difficulty = "medium"
        else: difficulty = "hard"
        logger.info(f"ML Prediction: Time={predicted_time:.2f}s, Diff='{difficulty}' for Params: {params} (Actual Word Rarity Val: {word_rarity_for_model_input})")
        return {'predicted_solve_time': round(predicted_time, 2), 'difficulty': difficulty, 'is_fallback': False}
    except Exception as e:
        logger.error(f"Error in ML predict_difficulty_for_params: {e}", exc_info=True)
        if actual_selected_words: params['selected_puzzle_words_for_ml'] = actual_selected_words
        return get_fallback_prediction(params)


def generate_solvable_puzzle(target_difficulty: str, user_performance_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    puzzle_id = str(uuid.uuid4())
    num_words_total = NUM_GROUPS * WORDS_PER_GROUP
    logger.info(f"\n--- Generating Puzzle. Target Diff: '{target_difficulty.upper()}' ---")
    if user_performance_summary: logger.info(f"User Perf for '{target_difficulty}': {user_performance_summary}")
    else: logger.info("No user performance summary for this generation.")

    generation_params, predicted_result, found_matching_params = {}, {}, False
    best_params_so_far, best_prediction_so_far = None, None
    param_candidates = []
    actual_search_difficulty = target_difficulty

    if user_performance_summary and user_performance_summary.get("plays", 0) >= 2: 
        win_rate = user_performance_summary.get("win_rate", 0.5); avg_hints = user_performance_summary.get("avg_hints", MAX_HINTS / 2.0)
        if target_difficulty=="easy" and win_rate>0.9 and avg_hints<0.2: actual_search_difficulty="medium"; logger.info("Nudge: Easy->Medium")
        elif target_difficulty=="medium":
            if win_rate>0.75 and avg_hints<0.5: actual_search_difficulty="hard"; logger.info("Nudge: Medium->Hard")
            elif win_rate<0.3 and avg_hints>(MAX_HINTS*0.66): actual_search_difficulty="easy"; logger.info("Nudge: Medium->Easy")
        elif target_difficulty=="hard" and win_rate<0.25 and avg_hints>(MAX_HINTS*0.66): actual_search_difficulty="medium"; logger.info("Nudge: Hard->Medium")
    
    if actual_search_difficulty == 'easy': param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd} for ct in ['same_category', 'begins_with'] for wr in ['very_common', 'common'] for sd in [random.uniform(1,3)]] * 8)
    elif actual_search_difficulty == 'medium': param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd} for ct in ['same_category', 'syllable_count', 'rhyming_words'] for wr in ['common', 'somewhat_common'] for sd in [random.uniform(3,6)]] * 8)
    else: param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd} for ct in ['conceptual_relation', 'synonym_groups', 'anagrams'] for wr in ['uncommon', 'rare'] for sd in [random.uniform(5,9)]] * 8)
    if not param_candidates: param_candidates = [{'connection_type': 'same_category', 'word_rarity': 'common', 'semantic_distance': 3.0}] * MAX_GENERATION_ATTEMPTS
    random.shuffle(param_candidates)

    for attempt in range(min(MAX_GENERATION_ATTEMPTS, len(param_candidates))):
        current_params = param_candidates[attempt]; current_params['num_words'] = num_words_total
        if current_params.get('connection_type') not in connection_types or current_params.get('word_rarity') not in word_rarity_levels: continue
        current_prediction = predict_difficulty_for_params(current_params) # Predict based on TARGET params first
        if not best_prediction_so_far or (not current_prediction.get('is_fallback') and best_prediction_so_far.get('is_fallback', True)):
            best_params_so_far = current_params.copy(); best_prediction_so_far = current_prediction.copy()
        if current_prediction.get('difficulty') == target_difficulty and not current_prediction.get('is_fallback'):
            generation_params = current_params; predicted_result = current_prediction; found_matching_params = True
            logger.info(f"--> Target '{target_difficulty}' MET by ML on attempt {attempt+1} with params: {generation_params}")
            break
    
    if not found_matching_params:
        logger.warning(f"No exact ML match for '{target_difficulty}'. Using best: {best_params_so_far} (Pred: {best_prediction_so_far})")
        if best_params_so_far and best_prediction_so_far: generation_params, predicted_result = best_params_so_far, best_prediction_so_far
        else:
            logger.error("No valid ML preds. Hardcoded fallback for gen."); generation_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total, 'semantic_distance': 3.0}
            predicted_result = get_fallback_prediction(generation_params)

    final_connection_type = generation_params.get('connection_type', 'same_category')
    logger.info(f"--- Finalizing words. Connection: '{final_connection_type}', TargetRarity: '{generation_params.get('word_rarity')}' ---")
    solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = {}, [], {}, {}
    
    word_selection_data = None
    if final_connection_type == 'same_category':
        word_selection_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    else: logger.warning(f"Word selection for '{final_connection_type}' not fully implemented. Using fallback.")

    if word_selection_data:
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = word_selection_data
    else: 
        logger.info(f"Specific selection failed/not impl for '{final_connection_type}'. Fallback for type '{final_connection_type}'.")
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

    # After words are selected, optionally re-predict difficulty based on *actual* words for more accuracy
    # This makes the generation loop more complex as prediction depends on word selection.
    # For now, we use the prediction based on target parameters.
    # If you wanted to do this:
    # final_prediction_with_actual_words = predict_difficulty_for_params(generation_params, actual_selected_words=all_words_for_grid)
    # actual_generated_difficulty = final_prediction_with_actual_words.get('difficulty', target_difficulty)
    # predicted_solve_time_final = final_prediction_with_actual_words.get('predicted_solve_time', -1.0)
    # is_fallback_final = final_prediction_with_actual_words.get('is_fallback', True)
    # Instead, we use the predicted_result from the parameter search.
    actual_generated_difficulty = predicted_result.get('difficulty', target_difficulty)
    predicted_solve_time_final = predicted_result.get('predicted_solve_time', -1.0)
    is_fallback_final = predicted_result.get('is_fallback', True)


    random.shuffle(all_words_for_grid)
    active_puzzles[puzzle_id] = {
        "puzzle_id": puzzle_id, "words_on_grid": [w.upper() for w in all_words_for_grid],
        "solution": {k: [w.upper() for w in v] for k,v in solution_groups.items()}, 
        "descriptions": group_descriptions, "difficulty": actual_generated_difficulty, # Use the potentially re-evaluated difficulty
        "predicted_solve_time": predicted_solve_time_final,
        "is_fallback_prediction": is_fallback_final,
        "parameters": {**generation_params, "difficulty_index_map": difficulty_index_map_for_puzzle},
        "creation_time": datetime.datetime.now(), "is_daily": False
    }
    logger.info(f"--- Puzzle {puzzle_id} (Target: {target_difficulty}, Final Gen Diff: {active_puzzles[puzzle_id]['difficulty']}) created. ---")
    return {"puzzle_id": puzzle_id, "words": active_puzzles[puzzle_id]["words_on_grid"], "difficulty": active_puzzles[puzzle_id]["difficulty"]}

def get_or_generate_daily_challenge() -> dict:
    # ... (same as before) ...
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
    # ... (same as before) ...
    if puzzle_id not in active_puzzles: return {"correct": False, "message": "Invalid puzzle.", "solved_groups": {}}
    data = active_puzzles[puzzle_id]; solution = data["solution"]; descriptions = data["descriptions"]
    diff_idx_map = data.get("parameters", {}).get("difficulty_index_map", {})
    # Check if puzzle_data has a set of already solved group keys
    server_solved_groups = data.get("_server_solved_group_keys", set())

    attempt_key = next(iter(user_groups_attempt), None)
    if not attempt_key: return {"correct": False, "message": "No attempt.", "solved_groups": {}}
    attempt_words = sorted([w.upper() for w in user_groups_attempt[attempt_key]])
    if len(attempt_words) != WORDS_PER_GROUP: return {"correct": False, "message": f"Select {WORDS_PER_GROUP} words.", "solved_groups": {}}

    for gk, correct_words in solution.items():
        if attempt_words == correct_words:
            if gk in server_solved_groups: # Group already solved by this player in this session
                return {"correct": False, # Technically not a "new" correct group
                        "message": f"You've already found the group: {descriptions.get(gk, 'Found')}!", 
                        "solved_groups": {}} # No new groups to report as solved
            
            # Mark as solved on server side for this puzzle instance
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
    
    # Use server-side tracked solved groups if available, otherwise trust client (less secure but simpler for now)
    server_solved_groups = puzzle_data.get("_server_solved_group_keys", set(solved_group_keys_from_client))


    unsolved_groups = {k: {"words_upper": v, "words_lower": [w.lower() for w in v], 
                           "description": actual_descriptions.get(k, "A Group")}
                       for k, v in actual_solution.items() if k not in server_solved_groups} # Use server_solved_groups
                       
    if not unsolved_groups: return {"hint": None, "message": "All groups solved!", "words": []}
    
    target_group_key = random.choice(list(unsolved_groups.keys()))
    target_group_info = unsolved_groups[target_group_key]
    hint_text = f"Hint for '{target_group_info['description']}'."
    words_to_highlight = []

    if w2v_model and GENSIM_AVAILABLE:
        try:
            anchor_words_lower = random.sample(target_group_info["words_lower"], k=min(len(target_group_info["words_lower"]), 1))
            vocab_source = w2v_model # Assumes w2v_model is KeyedVectors after load
            if hasattr(w2v_model, 'wv') and hasattr(w2v_model.wv, 'key_to_index'): # If full Word2Vec model was loaded
                vocab_source = w2v_model.wv
            
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
                    hint_text = f"This group relates to '{target_group_info['description']}'. Think about concepts around '{valid_anchor_words[0].upper()}' or ideas like '{found_w2v_hint_word_upper}' (which is not in this puzzle)."
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