import random
import uuid
import joblib
import pandas as pd
import os
import datetime as dt
import time
import logging
import json
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict

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
BEST_MODEL_NAME_FROM_TRAINING = "ridge_regression" # Should match your model file's base name
DIFFICULTY_MODEL_FILENAME = os.path.join(MODEL_DIR, f"wordlinks_{BEST_MODEL_NAME_FROM_TRAINING}_model.pkl")
FEATURE_LIST_FILENAME = os.path.join(MODEL_DIR, "feature_list.pkl")

WORD2VEC_ACTUAL_FILENAME = "glove.6B.100d.kv" # Should match your Word2Vec file
WORD2VEC_MODEL_PATH = os.path.join(MODEL_DIR, WORD2VEC_ACTUAL_FILENAME)

ENRICHED_VOCAB_PATH = os.path.join(MODEL_DIR, "enriched_word_data.json")

MAX_GENERATION_ATTEMPTS = 30
WORDS_PER_GROUP = 4
NUM_GROUPS = 4
MAX_PUZZLE_AGE_SECONDS = 3 * 3600
MAX_HINTS = 3

EASY_THRESHOLD = 110
MEDIUM_THRESHOLD = 170

# MODIFIED: Import for get_random_approved_user_puzzle from database.py
try:
    # Assuming database.py is in the same directory or Python path is configured
    from database import get_random_approved_user_puzzle
    DATABASE_AVAILABLE = True
    logger.info("Successfully imported 'get_random_approved_user_puzzle' from database module.")
except ImportError as e:
    logger.error(f"CRITICAL IMPORT ERROR in puzzle_logic: {e}. Could not import 'get_random_approved_user_puzzle' from database module. User-submitted puzzles will be disabled.")
    DATABASE_AVAILABLE = False
    # Define a stub function so the rest of the code doesn't break if import fails
    def get_random_approved_user_puzzle() -> Optional[Dict[str, Any]]:
        logger.warning("Stub get_random_approved_user_puzzle called; database module not available.")
        return None


# --- Load Difficulty Prediction ML Model Pipeline ---
model_pipeline = None
ALL_EXPECTED_INPUT_FEATURES = []
try:
    if not BEST_MODEL_NAME_FROM_TRAINING:
        logger.warning(f"BEST_MODEL_NAME_FROM_TRAINING is not set. ML for difficulty disabled.")
    elif not os.path.isdir(MODEL_DIR):
        logger.warning(f"Model directory '{MODEL_DIR}' does not exist. ML for difficulty disabled.")
    elif not os.path.exists(DIFFICULTY_MODEL_FILENAME):
        logger.warning(f"Difficulty model '{DIFFICULTY_MODEL_FILENAME}' not found. ML disabled.")
    elif not os.path.exists(FEATURE_LIST_FILENAME):
        logger.warning(f"Feature list '{FEATURE_LIST_FILENAME}' not found. ML disabled.")
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
    model_pipeline = None
    ALL_EXPECTED_INPUT_FEATURES = []

# --- Load Word2Vec Model ---
w2v_model = None
if GENSIM_AVAILABLE:
    try:
        if not WORD2VEC_ACTUAL_FILENAME:
             logger.warning(f"WORD2VEC_ACTUAL_FILENAME is not set. W2V hints fallback.")
        elif not os.path.exists(WORD2VEC_MODEL_PATH):
            logger.warning(f"Word2Vec model '{WORD2VEC_MODEL_PATH}' not found. W2V hints fallback.")
        else:
            logger.info(f"Loading Word2Vec model: {WORD2VEC_MODEL_PATH}...")
            if WORD2VEC_ACTUAL_FILENAME.endswith(".kv"):
                 w2v_model = KeyedVectors.load(WORD2VEC_MODEL_PATH)
            elif WORD2VEC_ACTUAL_FILENAME.endswith((".bin", ".wordvectors")):
                 w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=WORD2VEC_ACTUAL_FILENAME.endswith(".bin"))
            elif WORD2VEC_ACTUAL_FILENAME.endswith(".model"):
                 w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH).wv
            else:
                 w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=False)
            logger.info(f"Word2Vec model '{WORD2VEC_MODEL_PATH}' loaded.")
    except Exception as e:
        logger.error(f"Err loading W2V model: {e}. W2V hints fallback.", exc_info=True)
        w2v_model = None
else:
    logger.warning("Gensim not installed. Word2Vec features will be limited or disabled.")

ENRICHED_VOCABULARY: Dict[str, Dict[str, Any]] = {}
word_categories_hardcoded = {
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'brown', 'gray', 'teal', 'maroon', 'indigo', 'violet', 'cyan', 'magenta', 'lime', 'olive', 'gold', 'silver', 'bronze', 'ivory', 'beige'],
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'snake', 'eagle', 'shark', 'whale', 'dolphin', 'penguin', 'owl', 'bat', 'bee', 'ant'],
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach', 'kiwi', 'watermelon', 'cherry', 'pear', 'blueberry', 'raspberry', 'lemon', 'lime', 'plum', 'apricot', 'fig', 'date', 'guava'],
    'kitchen_items': ['knife', 'fork', 'spoon', 'plate', 'bowl', 'cup', 'pan', 'pot', 'oven', 'microwave', 'blender', 'toaster', 'kettle', 'grater', 'whisk', 'ladle', 'spatula', 'colander']
}
word_categories: Dict[str, List[str]] = {}

if os.path.exists(ENRICHED_VOCAB_PATH):
    try:
        with open(ENRICHED_VOCAB_PATH, 'r', encoding='utf-8') as f:
            ENRICHED_VOCABULARY = json.load(f)
        logger.info(f"Enriched vocabulary loaded from {ENRICHED_VOCAB_PATH} ({len(ENRICHED_VOCABULARY)} words).")
        temp_word_categories = defaultdict(list)
        for word, data in ENRICHED_VOCABULARY.items():
            category = data.get("category")
            if category:
                temp_word_categories[category].append(word.lower())
        valid_categories_from_enriched = {
            cat: words for cat, words in temp_word_categories.items() if len(words) >= WORDS_PER_GROUP
        }
        if valid_categories_from_enriched:
            word_categories = dict(valid_categories_from_enriched)
            logger.info(f"Word categories populated from enriched data ({len(word_categories)} valid categories).")
        else:
            logger.warning("No valid categories with enough words in enriched vocab; using hardcoded.")
            word_categories = word_categories_hardcoded
    except Exception as e:
        logger.error(f"Error loading or processing {ENRICHED_VOCAB_PATH}: {e}", exc_info=True)
        word_categories = word_categories_hardcoded
else:
    logger.warning(f"'{ENRICHED_VOCAB_PATH}' not found. Using hardcoded categories.")
    word_categories = word_categories_hardcoded

connection_types = {
    'same_category': 1, 'begins_with': 2, 'ends_with': 3, 'syllable_count': 4,
    'synonym_groups': 5, 'antonym_groups': 6, 'compound_words': 7, 'rhyming_words': 7,
    'conceptual_relation': 8, 'multiple_rules': 10, 'letter_pattern': 5, 'anagrams': 9,
    'homophones': 6, 'contains_substring': 4, 'metaphorical_relation': 9,
    'word_length': 3, 'user_submitted': 5 # Added user_submitted with an example complexity
}
connection_descriptions = {
    'same_category': "Same Category", 'begins_with': "Begin With Same Letter",
    'ends_with': "End With Same Letter", 'syllable_count': "Same Syllable Count",
    'synonym_groups': "Synonyms", 'antonym_groups': "Antonyms",
    'compound_words': "Compound Words", 'rhyming_words': "Rhyming Words",
    'conceptual_relation': "Conceptual Link", 'multiple_rules': "Multiple Connections",
    'letter_pattern': "Shared Letters", 'anagrams': "Anagrams",
    'homophones': "Homophones", 'contains_substring': "Contain Substring",
    'metaphorical_relation': "Metaphorical Link",
    'word_length': "Same Word Length", 'user_submitted': "Community Puzzle" # Added description
}
word_rarity_levels = { 'very_common': 1, 'common': 2, 'somewhat_common': 4, 'uncommon': 6, 'rare': 8, 'very_rare': 10, 'extremely_rare': 12 }
active_puzzles: Dict[str, Dict[str, Any]] = {}

def _get_all_available_words(exclude_words: Optional[Set[str]] = None) -> List[str]:
    if exclude_words is None: exclude_words = set()
    all_words_from_vocab = set()
    if ENRICHED_VOCABULARY:
        for word in ENRICHED_VOCABULARY.keys():
            all_words_from_vocab.add(word.lower())
    else:
        for cat_words_list in word_categories.values():
            for word in cat_words_list:
                all_words_from_vocab.add(word.lower())
    return list(all_words_from_vocab - exclude_words)

def _get_words_by_target_rarity(source_word_list: List[str], target_rarity_name: str, count: int, exclude_list: Set[str]) -> List[str]:
    if not ENRICHED_VOCABULARY:
        available_words = [w.lower() for w in source_word_list if w.lower() not in exclude_list]
        return random.sample(available_words, min(count, len(available_words))) if len(available_words) >= count else []
    rarity_to_complexity_map = {
        'very_common': (1, 2), 'common': (3, 4), 'somewhat_common': (5, 6),
        'uncommon': (7, 8), 'rare': (9, 9), 'very_rare': (10, 10), 'extremely_rare': (10,10)
    }
    min_c, max_c = rarity_to_complexity_map.get(target_rarity_name, (1, 10))
    candidate_words = []
    for word_str in source_word_list:
        word_lower = word_str.lower()
        if word_lower not in exclude_list:
            word_data = ENRICHED_VOCABULARY.get(word_lower, {})
            complexity_score = word_data.get("complexity_score", 5)
            if min_c <= complexity_score <= max_c:
                candidate_words.append(word_lower)
    if len(candidate_words) >= count:
        return random.sample(candidate_words, count)
    else:
        logger.debug(f"Rarity '{target_rarity_name}' fallback for list of {len(source_word_list)} words. Found {len(candidate_words)}, need {count}.")
        available_fallback = [w.lower() for w in source_word_list if w.lower() not in exclude_list]
        return random.sample(available_fallback, min(count, len(available_fallback))) if len(available_fallback) >= count else []

def _generate_groups_by_category(num_groups_to_gen: int, words_per_group_val: int, used_words_global: Set[str], generation_params: Dict[str, Any]) -> Optional[Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]]:
    # ... (same as your last complete version) ...
    target_word_rarity = generation_params.get('word_rarity', 'common')
    all_cat_names = list(word_categories.keys())
    if not all_cat_names:
        logger.error("CatGen: No categories available in word_categories.")
        return None
    if len(all_cat_names) < num_groups_to_gen:
        logger.warning(f"CatGen: Not enough distinct categories ({len(all_cat_names)}) for {num_groups_to_gen} groups. Will use categories with replacement if needed.")
        if not all_cat_names: return None
        selected_cat_names = random.choices(all_cat_names, k=num_groups_to_gen)
    else:
        selected_cat_names = random.sample(all_cat_names, num_groups_to_gen)
    solution_groups, all_words_list, descriptions, diff_indices = {}, [], {}, {}
    current_puzzle_used_words = set(used_words_global)
    for i, cat_name in enumerate(selected_cat_names):
        group_id = f"group_{i+1}"
        words_in_category = word_categories.get(cat_name, [])
        group_words = _get_words_by_target_rarity(words_in_category, target_word_rarity, words_per_group_val, current_puzzle_used_words)
        if len(group_words) < words_per_group_val:
            logger.warning(f"CatGen: Failed for cat '{cat_name}', rarity '{target_word_rarity}'. Found {len(group_words)} words. Aborting.")
            return None
        solution_groups[group_id] = sorted(group_words)
        all_words_list.extend(group_words)
        current_puzzle_used_words.update(group_words)
        descriptions[group_id] = connection_descriptions.get('same_category', "Category") + f": {cat_name.replace('_', ' ').capitalize()}"
        diff_indices[group_id] = i
    if len(solution_groups) < num_groups_to_gen : return None
    return solution_groups, all_words_list, descriptions, diff_indices


def _generate_groups_by_word_length(num_groups_to_gen: int, words_per_group_val: int, used_words_global: Set[str], generation_params: Dict[str, Any]) -> Optional[Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]]:
    # ... (same as your last complete version) ...
    logger.info(f"Attempting to generate {num_groups_to_gen} groups by word length.")
    target_word_rarity = generation_params.get('word_rarity', 'common')
    all_available_words_for_this_type = _get_all_available_words(exclude_words=used_words_global)
    if len(all_available_words_for_this_type) < num_groups_to_gen * words_per_group_val:
        logger.warning("WordLengthGen: Not enough available words overall.")
        return None
    words_by_length: Dict[int, List[str]] = defaultdict(list)
    for word in all_available_words_for_this_type: words_by_length[len(word)].append(word)
    MIN_WORD_LENGTH_FOR_GROUP = 3; MAX_WORD_LENGTH_FOR_GROUP = 10
    potential_lengths = [l for l, wl in words_by_length.items() if len(wl) >= words_per_group_val and MIN_WORD_LENGTH_FOR_GROUP <= l <= MAX_WORD_LENGTH_FOR_GROUP]
    if len(potential_lengths) < num_groups_to_gen:
        logger.warning(f"WordLengthGen: Not enough distinct lengths. Found {len(potential_lengths)}, need {num_groups_to_gen}.")
        return None
    selected_lengths_for_groups = random.sample(potential_lengths, num_groups_to_gen)
    solution_groups, all_words_list, descriptions, diff_indices = {}, [], {}, {}
    current_puzzle_used_words = set(used_words_global)
    for i, length in enumerate(selected_lengths_for_groups):
        group_id = f"group_{i+1}"
        words_of_specific_length = words_by_length[length]
        group_words = _get_words_by_target_rarity(words_of_specific_length, target_word_rarity, words_per_group_val, current_puzzle_used_words)
        if len(group_words) < words_per_group_val:
            logger.warning(f"WordLengthGen: Failed for length {length}, rarity '{target_word_rarity}'. Found {len(group_words)}.")
            return None
        solution_groups[group_id] = sorted(group_words)
        all_words_list.extend(group_words)
        current_puzzle_used_words.update(group_words)
        descriptions[group_id] = f"{length}-Letter Words"
        diff_indices[group_id] = i
    if len(solution_groups) == num_groups_to_gen:
        logger.info(f"Successfully generated {num_groups_to_gen} groups by word length.")
        return solution_groups, all_words_list, descriptions, diff_indices
    return None

def _generate_groups_by_starting_letter(num_groups_to_gen: int, words_per_group_val: int, used_words_global: Set[str], generation_params: Dict[str, Any]) -> Optional[Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]]:
    # ... (same as your last complete version, with target_word_rarity defined) ...
    logger.info(f"Attempting to generate {num_groups_to_gen} groups by starting letter.")
    target_word_rarity = generation_params.get('word_rarity', 'common')
    all_available_words_for_this_type = _get_all_available_words(exclude_words=used_words_global)
    if len(all_available_words_for_this_type) < num_groups_to_gen * words_per_group_val:
        logger.warning("StartLetterGen: Not enough available words overall.")
        return None
    words_by_letter: Dict[str, List[str]] = defaultdict(list)
    for word in all_available_words_for_this_type: words_by_letter[word[0].lower()].append(word)
    potential_letters = [l for l, wl in words_by_letter.items() if len(wl) >= words_per_group_val]
    if len(potential_letters) < num_groups_to_gen:
        logger.warning(f"StartLetterGen: Not enough distinct starting letters. Found {len(potential_letters)}, need {num_groups_to_gen}.")
        return None
    selected_letters_for_groups = random.sample(potential_letters, num_groups_to_gen)
    solution_groups, all_words_list, descriptions, diff_indices = {}, [], {}, {}
    current_puzzle_used_words = set(used_words_global)
    for i, letter in enumerate(selected_letters_for_groups):
        group_id = f"group_{i+1}"
        words_starting_with_letter = words_by_letter[letter]
        group_words = _get_words_by_target_rarity(words_starting_with_letter, target_word_rarity, words_per_group_val, current_puzzle_used_words)
        if len(group_words) < words_per_group_val:
            logger.warning(f"StartLetterGen: Failed for letter '{letter}', rarity '{target_word_rarity}'. Found {len(group_words)}.")
            return None
        solution_groups[group_id] = sorted(group_words)
        all_words_list.extend(group_words)
        current_puzzle_used_words.update(group_words)
        descriptions[group_id] = f"Words Starting With '{letter.upper()}'"
        diff_indices[group_id] = i
    if len(solution_groups) == num_groups_to_gen:
        logger.info(f"Successfully generated {num_groups_to_gen} groups by starting letter.")
        return solution_groups, all_words_list, descriptions, diff_indices
    return None


def _generate_groups_by_ending_letter(num_groups: int, words_per_group: int, used_words: Set[str], params: Dict[str, Any]) -> Optional[Tuple[Dict, List, Dict, Dict]]:
    logger.warning(f"Word selection logic for 'ends_with' is a STUB. Using fallback.")
    return None
def _generate_groups_by_rhyme(num_groups: int, words_per_group: int, used_words: Set[str], params: Dict[str, Any]) -> Optional[Tuple[Dict, List, Dict, Dict]]:
    logger.warning(f"Word selection logic for 'rhyming_words' is a STUB. Using fallback.")
    return None
def _generate_groups_by_anagram(num_groups: int, words_per_group: int, used_words: Set[str], params: Dict[str, Any]) -> Optional[Tuple[Dict, List, Dict, Dict]]:
    logger.warning(f"Word selection logic for 'anagrams' is a STUB. Using fallback.")
    return None

def _generate_fallback_groups(num_words_needed_val: int, original_connection_type: str, puzzle_params_for_fallback: Dict[str, Any]) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]:
    # ... (same as before) ...
    logger.warning(f"[FALLBACK-GROUPS] Using 'same_category' logic as fallback for original request '{original_connection_type}'.")
    fallback_params = {**puzzle_params_for_fallback, 'word_rarity': puzzle_params_for_fallback.get('word_rarity', 'common')}
    category_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set(), fallback_params)
    if category_data: return category_data
    else:
        logger.critical("CRITICAL FALLBACK: _generate_groups_by_category FAILED. Using dummy.")
        dummy_solution = {f"g{i+1}": [f"err_w{i*WORDS_PER_GROUP+j+1}" for j in range(WORDS_PER_GROUP)] for i in range(NUM_GROUPS)}
        dummy_words = [w for grp in dummy_solution.values() for w in grp]
        return dummy_solution, dummy_words, {k:"Error Group" for k in dummy_solution}, {k:i for i,k in enumerate(dummy_solution)}

def get_fallback_prediction(params: Dict[str, Any], actual_selected_words: Optional[List[str]] = None) -> Dict[str, Any]:
    # ... (same as before) ...
    complexity = connection_types.get(params.get('connection_type', 'same_category'), 1)
    rarity_score_to_use = word_rarity_levels.get(params.get('word_rarity', 'common'), 2)
    if actual_selected_words and ENRICHED_VOCABULARY:
        complexities = [ENRICHED_VOCABULARY.get(w.lower(), {}).get('complexity_score', 5) for w in actual_selected_words]
        if complexities:
            avg_word_complexity_score = sum(complexities) / len(complexities)
            if avg_word_complexity_score <= 2.5: rarity_score_to_use = word_rarity_levels['very_common']
            elif avg_word_complexity_score <= 4.5: rarity_score_to_use = word_rarity_levels['common']
            elif avg_word_complexity_score <= 6.5: rarity_score_to_use = word_rarity_levels['somewhat_common']
            elif avg_word_complexity_score <= 8.5: rarity_score_to_use = word_rarity_levels['uncommon']
            else: rarity_score_to_use = word_rarity_levels['rare']
    score = complexity + rarity_score_to_use
    est_time = 20 + score * 5 + (params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP) - (NUM_GROUPS * WORDS_PER_GROUP)) * 0.5
    difficulty = "medium"
    if est_time <= EASY_THRESHOLD * 0.8: difficulty = "easy"
    elif est_time <= MEDIUM_THRESHOLD * 1.1: difficulty = "medium"
    else: difficulty = "hard"
    logger.info(f"[FALLBACK-PRED] Rule-based: ConnComp={complexity}, RarityScore={rarity_score_to_use}, EstTime={est_time:.1f}s, Diff='{difficulty}' for params: {params.get('connection_type')}/{params.get('word_rarity')}")
    return {'predicted_solve_time': round(est_time, 2), 'difficulty': difficulty, 'is_fallback': True}

def predict_difficulty_for_params(params: Dict[str, Any], actual_selected_words: Optional[List[str]] = None) -> Dict[str, Any]:
    # ... (same as before) ...
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
        feature_data_dict.update({
            'num_words': float(params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP)),
            'connection_complexity': float(connection_types.get(params.get('connection_type'), 5)),
            'word_rarity_value': float(word_rarity_for_model_input),
            'semantic_distance': float(params.get('semantic_distance', 5.0)),
            'time_of_day': float(dt.datetime.now().hour),
            'hints_used': 0.0, 'num_players': 50.0, 'completions': 40.0,
            'completion_rate': 0.80, 'attempt_count': 2.0, 'time_before_first_attempt': 10.0,
            'hover_count': float(params.get('num_words', 16) * 1.5), 'abandonment_rate': 0.20,
            'competitiveness_score': 5.0, 'frustration_score': 3.0, 'satisfaction_score': 7.0,
            'learning_value': 5.0, 'engagement_score': 6.0, 'replayability_score': 4.0,
            'avg_attempts_before_success': 1.5
        })
        final_feature_values = [feature_data_dict.get(name, 0.0) for name in ALL_EXPECTED_INPUT_FEATURES]
        if len(final_feature_values) != len(ALL_EXPECTED_INPUT_FEATURES):
            logger.error(f"Feature length mismatch. Expected {len(ALL_EXPECTED_INPUT_FEATURES)}, got {len(final_feature_values)}")
            return get_fallback_prediction(params, actual_selected_words)
        predict_df = pd.DataFrame([final_feature_values], columns=ALL_EXPECTED_INPUT_FEATURES)
        predicted_time = float(model_pipeline.predict(predict_df)[0])
        difficulty = "hard"
        if predicted_time < EASY_THRESHOLD: difficulty = "easy"
        elif predicted_time < MEDIUM_THRESHOLD: difficulty = "medium"
        logger.info(f"ML Prediction: Time={predicted_time:.2f}s, Diff='{difficulty}' for Params: {params.get('connection_type')}/{params.get('word_rarity')} (RarityVal for Model: {word_rarity_for_model_input})")
        return {'predicted_solve_time': round(predicted_time, 2), 'difficulty': difficulty, 'is_fallback': False}
    except Exception as e:
        logger.error(f"Error in ML predict_difficulty_for_params: {e}", exc_info=True)
        return get_fallback_prediction(params, actual_selected_words)

def generate_solvable_puzzle(target_difficulty: str, user_performance_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    puzzle_id_base = str(uuid.uuid4())
    num_words_total = NUM_GROUPS * WORDS_PER_GROUP
    logger.info(f"\n--- Generating Puzzle. Target Diff: '{target_difficulty.upper()}' ---")
    if user_performance_summary: logger.info(f"User Perf for '{target_difficulty}': {user_performance_summary}")
    else: logger.info("No user performance summary for this generation.")

    # --- MODIFIED: Chance to use a user-submitted puzzle ---
    USE_USER_PUZZLE_CHANCE = 0.25 # Example: 25% chance
    if DATABASE_AVAILABLE and random.random() < USE_USER_PUZZLE_CHANCE: # Only if DB function is available
        logger.info(f"Attempting to select an approved user-submitted puzzle (Chance: {USE_USER_PUZZLE_CHANCE*100}%).")
        user_puzzle_data_from_db = get_random_approved_user_puzzle() # Function from database.py
        if user_puzzle_data_from_db:
            logger.info(f"Selected user puzzle ID (from DB): {user_puzzle_data_from_db['id']}")
            puzzle_id = f"user_{user_puzzle_data_from_db['id']}_{puzzle_id_base[:8]}"

            solution_groups = {}
            all_words_list = []
            group_descriptions = {}
            temp_difficulty_index_map = {}

            valid_puzzle_structure = True
            for i in range(1, 5):
                group_key = f"group_{i}"
                words_for_group_json = user_puzzle_data_from_db.get(f'group{i}_words')
                if not words_for_group_json: # Should not happen if data is validated on submission
                    logger.error(f"User puzzle ID {user_puzzle_data_from_db['id']} group {i} has no words. Skipping.")
                    valid_puzzle_structure = False; break
                
                try: # Words are stored as JSON strings in DB
                    words_for_group = json.loads(words_for_group_json)
                except json.JSONDecodeError:
                    logger.error(f"User puzzle ID {user_puzzle_data_from_db['id']} group {i} words are not valid JSON. Skipping.")
                    valid_puzzle_structure = False; break

                processed_words = sorted(list(set(w.upper() for w in words_for_group if isinstance(w, str) and w.strip())))
                if len(processed_words) != WORDS_PER_GROUP:
                    logger.error(f"User puzzle ID {user_puzzle_data_from_db['id']} group {i} does not have {WORDS_PER_GROUP} valid unique words after processing. Skipping.")
                    valid_puzzle_structure = False; break
                
                solution_groups[group_key] = processed_words
                all_words_list.extend(processed_words)
                group_descriptions[group_key] = user_puzzle_data_from_db.get(f'group{i}_description', f"Community Group {i}")
                temp_difficulty_index_map[group_key] = i -1

            if not valid_puzzle_structure:
                logger.warning("Failed to process user puzzle structure, falling back to algorithmic generation.")
                # Fall through to regular generation by not returning here
            elif len(set(all_words_list)) != num_words_total:
                logger.error(f"User puzzle ID {user_puzzle_data_from_db['id']} does not contain {num_words_total} unique words overall. Skipping.")
                # Fall through to regular generation
            else:
                user_puzzle_gen_params = {
                    'connection_type': 'user_submitted',
                    'word_rarity': 'mixed',
                    'num_words': num_words_total,
                    'semantic_distance': 5.0
                }
                if 'user_submitted' not in connection_types: connection_types['user_submitted'] = 5 # Default complexity
                if 'user_submitted' not in connection_descriptions: connection_descriptions['user_submitted'] = "Community Puzzle"


                final_puzzle_difficulty_data = predict_difficulty_for_params(user_puzzle_gen_params, actual_selected_words=all_words_list)
                display_difficulty = f"Community Puzzle (by {user_puzzle_data_from_db.get('submitter_name', 'Anonymous') or 'Anonymous'})"
                ml_predicted_difficulty = final_puzzle_difficulty_data.get('difficulty', 'medium')

                random.shuffle(all_words_list) # Shuffle for grid display
                active_puzzles[puzzle_id] = {
                    "puzzle_id": puzzle_id, "words_on_grid": all_words_list,
                    "solution": solution_groups, "descriptions": group_descriptions,
                    "difficulty": display_difficulty,
                    "parameters": { **user_puzzle_gen_params,
                                   "original_submitter": user_puzzle_data_from_db.get('submitter_name', 'Anonymous'),
                                   "original_db_id": user_puzzle_data_from_db['id'],
                                   "ml_predicted_difficulty": ml_predicted_difficulty,
                                   "difficulty_index_map": temp_difficulty_index_map },
                    "predicted_solve_time": final_puzzle_difficulty_data.get('predicted_solve_time', 180),
                    "is_fallback_prediction": final_puzzle_difficulty_data.get('is_fallback', True),
                    "creation_time": dt.datetime.now(), "is_daily": False,
                    "_server_solved_group_keys": set()
                }
                logger.info(f"--- Using User Puzzle {puzzle_id} (DB ID: {user_puzzle_data_from_db['id']}) created. ---")
                return {"puzzle_id": puzzle_id, "words": active_puzzles[puzzle_id]["words_on_grid"], "difficulty": display_difficulty}
        elif DATABASE_AVAILABLE and USE_USER_PUZZLE_CHANCE > 0:
             logger.info("No approved user puzzle found or chance not met, proceeding with algorithmic generation.")
        elif not DATABASE_AVAILABLE and USE_USER_PUZZLE_CHANCE > 0:
            logger.warning("Database module/function not available, cannot fetch user puzzles.")


    # --- REGULAR ALGORITHMIC PUZZLE GENERATION ---
    generation_params, predicted_result_for_params, found_matching_params = {}, {}, False
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

    if actual_search_difficulty == 'easy':
        param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd}
            for ct in ['same_category', 'begins_with', 'word_length', 'contains_substring']
            for wr in ['very_common', 'common']
            for sd in [random.uniform(1,2.5), random.uniform(2,4)]] * 5)
    elif actual_search_difficulty == 'medium':
        param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd}
            for ct in ['same_category', 'begins_with', 'word_length', 'syllable_count', 'letter_pattern', 'homophones']
            for wr in ['common', 'somewhat_common', 'uncommon']
            for sd in [random.uniform(3,5), random.uniform(4.5,7)]] * 4)
    else:
        param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd}
            for ct in ['conceptual_relation', 'synonym_groups', 'antonym_groups', 'anagrams', 'metaphorical_relation', 'multiple_rules', 'word_length']
            for wr in ['uncommon', 'rare', 'very_rare']
            for sd in [random.uniform(5.5,7.5), random.uniform(7,9.5)]] * 3)
    if not param_candidates:
        param_candidates = [{'connection_type': 'same_category', 'word_rarity': 'common', 'semantic_distance': 3.0}] * MAX_GENERATION_ATTEMPTS
    random.shuffle(param_candidates)

    for attempt in range(min(MAX_GENERATION_ATTEMPTS, len(param_candidates))):
        current_params = param_candidates[attempt].copy()
        current_params['num_words'] = num_words_total
        if current_params.get('connection_type') not in connection_types or current_params.get('word_rarity') not in word_rarity_levels: continue
        current_prediction = predict_difficulty_for_params(current_params)
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
            break
    if not found_matching_params:
        logger.warning(f"No exact ML match for '{target_difficulty}'. Using best found: {best_params_so_far} (Pred: {best_prediction_so_far})")
        if best_params_so_far and best_prediction_so_far:
            generation_params = best_params_so_far
            predicted_result_for_params = best_prediction_so_far
        else:
            logger.error("No valid ML preds. Hardcoded fallback params for generation.")
            generation_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total, 'semantic_distance': 3.0}
            predicted_result_for_params = get_fallback_prediction(generation_params)

    final_connection_type = generation_params.get('connection_type', 'same_category')
    logger.info(f"--- Finalizing words (algorithmic). Connection: '{final_connection_type}', TargetRarity: '{generation_params.get('word_rarity')}' ---")
    word_selection_data = None
    if final_connection_type == 'same_category':
        word_selection_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'begins_with':
        word_selection_data = _generate_groups_by_starting_letter(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'word_length':
        word_selection_data = _generate_groups_by_word_length(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'ends_with':
        word_selection_data = _generate_groups_by_ending_letter(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'rhyming_words':
        word_selection_data = _generate_groups_by_rhyme(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'anagrams':
        word_selection_data = _generate_groups_by_anagram(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    else: logger.warning(f"Word selection logic for '{final_connection_type}' is a STUB.")

    if not word_selection_data:
        logger.info(f"Algorithmic selection failed for '{final_connection_type}'. Using fallback groups.")
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
            _generate_fallback_groups(num_words_total, final_connection_type, generation_params)
    else:
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = word_selection_data

    if not all_words_for_grid or len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
        logger.critical(f"CRITICAL: Algorithmic puzzle assembly error. Emergency Fallback.")
        emergency_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total}
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
            _generate_fallback_groups(num_words_total, "same_category", emergency_params)
        if not all_words_for_grid or len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
             raise ValueError("Emergency fallback also failed.")

    final_puzzle_difficulty_data = predict_difficulty_for_params(generation_params, actual_selected_words=all_words_for_grid)
    puzzle_id = f"algo_{puzzle_id_base[:8]}" # Different prefix for algorithmic puzzles

    random.shuffle(all_words_for_grid)
    active_puzzles[puzzle_id] = {
        "puzzle_id": puzzle_id, "words_on_grid": [w.upper() for w in all_words_for_grid],
        "solution": {k: sorted([w.upper() for w in v]) for k,v in solution_groups.items()},
        "descriptions": group_descriptions,
        "difficulty": final_puzzle_difficulty_data.get('difficulty', target_difficulty),
        "predicted_solve_time": final_puzzle_difficulty_data.get('predicted_solve_time', -1.0),
        "is_fallback_prediction": final_puzzle_difficulty_data.get('is_fallback', True),
        "parameters": {**generation_params, "difficulty_index_map": difficulty_index_map_for_puzzle},
        "creation_time": dt.datetime.now(), "is_daily": False,
        "_server_solved_group_keys": set()
    }
    logger.info(f"--- Algorithmic Puzzle {puzzle_id} (Target: {target_difficulty}, Conn: {final_connection_type}, Final Diff: {active_puzzles[puzzle_id]['difficulty']}) created. ---")
    return {"puzzle_id": puzzle_id, "words": active_puzzles[puzzle_id]["words_on_grid"], "difficulty": active_puzzles[puzzle_id]["difficulty"]}


# ... (get_or_generate_daily_challenge, check_puzzle_answer, get_puzzle_hint, cleanup_old_puzzles - same as before)
# Ensure their logic is compatible with the potential for user-submitted puzzle structures if needed,
# though generally they operate on the `active_puzzles` structure which should be consistent.

def get_or_generate_daily_challenge() -> dict: # ... (No changes from your last complete version)
    cleanup_old_puzzles()
    today_str = dt.date.today().isoformat() # Use dt for datetime module to avoid conflict
    daily_puzzle_id = f"daily_{today_str}"
    if daily_puzzle_id not in active_puzzles:
        logger.info(f"Generating new daily challenge for {today_str}")
        original_random_state = random.getstate()
        random.seed(today_str)
        temp_data = generate_solvable_puzzle(target_difficulty="medium")
        temp_id = temp_data["puzzle_id"]
        if temp_id in active_puzzles:
            daily_details = active_puzzles.pop(temp_id)
            daily_details.update({
                "puzzle_id": daily_puzzle_id, "is_daily": True, "difficulty": "Daily Challenge"
            })
            active_puzzles[daily_puzzle_id] = daily_details
            logger.info(f"Daily challenge {daily_puzzle_id} created and stored.")
        else:
            random.setstate(original_random_state)
            logger.critical(f"Failed to find temp puzzle {temp_id} for daily generation.")
            raise ValueError("Daily challenge temp puzzle error.")
        random.setstate(original_random_state)
    else:
        logger.info(f"Returning existing daily challenge for {today_str}: {daily_puzzle_id}")
    puzzle = active_puzzles[daily_puzzle_id]
    return {"puzzle_id": puzzle["puzzle_id"], "words": puzzle["words_on_grid"],
            "difficulty": puzzle["difficulty"], "is_daily": True }


def check_puzzle_answer(puzzle_id: str, user_groups_attempt: Dict[str, List[str]]) -> Dict[str, Any]: # ... (No changes)
    if puzzle_id not in active_puzzles:
        return {"correct": False, "message": "Invalid or expired puzzle ID.", "solved_groups": {}}
    data = active_puzzles[puzzle_id]
    solution = data.get("solution", {})
    descriptions = data.get("descriptions", {})
    diff_idx_map = data.get("parameters", {}).get("difficulty_index_map", {})
    server_solved_groups = data.setdefault("_server_solved_group_keys", set())
    attempt_key = next(iter(user_groups_attempt), None)
    if not attempt_key:
        return {"correct": False, "message": "No attempt data provided.", "solved_groups": {}}
    attempt_words = sorted([w.upper() for w in user_groups_attempt[attempt_key]])
    if len(attempt_words) != WORDS_PER_GROUP:
        return {"correct": False, "message": f"Please select exactly {WORDS_PER_GROUP} words.", "solved_groups": {}}
    for group_key_solution, correct_words_solution in solution.items():
        if attempt_words == correct_words_solution:
            if group_key_solution in server_solved_groups:
                return {"correct": False, "message": f"The group '{descriptions.get(group_key_solution, 'Group')}' has already been solved.", "solved_groups": {}}
            server_solved_groups.add(group_key_solution)
            logger.info(f"Correct group '{group_key_solution}' found for puzzle {puzzle_id}.")
            return {"correct": True, "message": f"Correct! You found: {descriptions.get(group_key_solution, 'a group')}",
                    "solved_groups": {group_key_solution: {"description": descriptions.get(group_key_solution, "Unknown Category"),
                                               "difficulty_index": diff_idx_map.get(group_key_solution,0)}}}
    logger.info(f"Incorrect group attempt for puzzle {puzzle_id}: {attempt_words}")
    return {"correct": False, "message": "That's not one of the groups. Try again!", "solved_groups": {}}

def get_puzzle_hint(puzzle_id: str, solved_group_keys_from_client: Optional[List[str]] = None) -> Dict[str, Any]: # ... (No changes)
    if solved_group_keys_from_client is None: solved_group_keys_from_client = []
    if puzzle_id not in active_puzzles:
        return {"hint": None, "message": "Invalid or expired puzzle for hint.", "words": []}
    puzzle_data = active_puzzles[puzzle_id]
    actual_solution = puzzle_data.get("solution", {})
    actual_descriptions = puzzle_data.get("descriptions", {})
    all_words_on_grid_upper = set(puzzle_data.get("words_on_grid", []))
    server_solved_groups = puzzle_data.get("_server_solved_group_keys", set())
    combined_solved_keys = server_solved_groups.union(set(solved_group_keys_from_client))
    unsolved_groups_details = {
        k: {"words_upper": v, "words_lower": [w.lower() for w in v], "description": actual_descriptions.get(k, "A Group")}
        for k, v in actual_solution.items() if k not in combined_solved_keys
    }
    if not unsolved_groups_details:
        return {"hint": None, "message": "Congratulations, all groups have been solved!", "words": []}
    target_group_key = random.choice(list(unsolved_groups_details.keys()))
    target_group_info = unsolved_groups_details[target_group_key]
    hint_text = f"Hint for the group: '{target_group_info['description']}'."
    words_to_highlight_on_grid = []
    if w2v_model and GENSIM_AVAILABLE:
        try:
            words_from_group_in_vocab = [w for w in target_group_info["words_lower"] if w in w2v_model]
            if words_from_group_in_vocab:
                anchor_words_for_w2v = random.sample(words_from_group_in_vocab, k=min(len(words_from_group_in_vocab), 2))
                similar_candidates = w2v_model.most_similar(positive=anchor_words_for_w2v, topn=20)
                found_external_hint_word = None
                for sim_word_lower, _ in similar_candidates:
                    sim_word_upper = sim_word_lower.upper()
                    is_puzzle_word = sim_word_upper in all_words_on_grid_upper
                    is_in_any_solution_group = any(sim_word_upper in sol_group for sol_group in actual_solution.values())
                    if sim_word_lower.isalpha() and len(sim_word_lower) > 2 and not is_puzzle_word and not is_in_any_solution_group:
                        found_external_hint_word = sim_word_lower
                        break
                if found_external_hint_word:
                    hint_text = f"The group '{target_group_info['description']}' might relate to concepts like '{found_external_hint_word.capitalize()}' (this word is not in the puzzle)."
                    words_to_highlight_on_grid.append(random.choice(anchor_words_for_w2v).upper())
                else:
                    logger.info("W2V: No suitable distinct external hint word found for group '%s'.", target_group_key)
            else:
                logger.info("W2V: None of the words from group '%s' are in the Word2Vec vocabulary.", target_group_key)
        except Exception as e:
            logger.error(f"Error during Word2Vec hint generation for group '%s': {e}", target_group_key, exc_info=True)
    if not words_to_highlight_on_grid:
        logger.info("Using basic hint (revealing words from group '%s').", target_group_key)
        num_to_reveal = random.randint(1, min(len(target_group_info["words_upper"]), 2))
        words_to_highlight_on_grid = random.sample(target_group_info["words_upper"], num_to_reveal)
        if words_to_highlight_on_grid:
            hint_text = f"Hint for '{target_group_info['description']}': This group includes {', '.join(words_to_highlight_on_grid)}."
        else:
            hint_text = f"Try to find words that fit the category: '{target_group_info['description']}'."
    return {"hint": hint_text, "words": words_to_highlight_on_grid, "message": "Hint provided."}

def cleanup_old_puzzles(): # ... (No changes from your last complete version)
    current_time_dt = dt.datetime.now()
    today_date = dt.date.today()
    puzzles_to_delete = []
    for pid, data in list(active_puzzles.items()):
        creation_time_dt = data.get("creation_time")
        if not isinstance(creation_time_dt, dt.datetime):
            if isinstance(creation_time_dt, (float, int)):
                try: creation_time_dt = dt.datetime.fromtimestamp(creation_time_dt)
                except (OSError, TypeError, ValueError) as ts_err:
                    logger.warning(f"Puzzle {pid} invalid timestamp {data.get('creation_time')}. Error: {ts_err}. Skipping."); continue
            else:
                logger.warning(f"Puzzle {pid} invalid creation_time type {type(creation_time_dt)}. Skipping."); continue
        if pid.startswith("daily_"):
            try:
                puzzle_date_str = pid.replace("daily_", "")
                if dt.date.fromisoformat(puzzle_date_str) < today_date:
                    puzzles_to_delete.append(pid)
            except ValueError: logger.error(f"Malformed daily puzzle ID for cleanup: {pid}.")
        elif (current_time_dt - creation_time_dt).total_seconds() > MAX_PUZZLE_AGE_SECONDS:
            puzzles_to_delete.append(pid)
    for pid in puzzles_to_delete:
        if pid in active_puzzles:
            del active_puzzles[pid]
            logger.info(f"Cleaned up old puzzle: {pid}")

if __name__ == "__main__":
    # ... (same as your last complete version, ensure it calls the functions you want to test) ...
    logger.info("Testing puzzle_logic.py standalone functions...")
    if not model_pipeline: logger.warning("ML Model not loaded for standalone test, expect fallback predictions.")
    if not w2v_model: logger.warning("Word2Vec Model not loaded for standalone test, expect fallback hints.")
    if not DATABASE_AVAILABLE: logger.warning("Database module not fully imported, user puzzle features in test might fail.")


    mock_good_perf_medium = {"plays": 5, "win_rate": 0.9, "avg_hints": 0.1, "avg_mistakes": 0.5, "avg_solve_time": 60}
    mock_struggling_perf_hard = {"plays": 5, "win_rate": 0.2, "avg_hints": 2.5, "avg_mistakes": 3, "avg_solve_time": 200}

    # Test algorithmic generation
    for diff_test in ["easy", "medium", "hard"]:
        try:
            print(f"\n--- Testing Algorithmic Puzzle Generation ({diff_test.capitalize()}) ---")
            perf_summary = None
            if diff_test == "medium": perf_summary = mock_good_perf_medium
            elif diff_test == "hard": perf_summary = mock_struggling_perf_hard
            
            # Temporarily set USE_USER_PUZZLE_CHANCE to 0 for this specific test block
            # This requires making USE_USER_PUZZLE_CHANCE a global or passing it,
            # or more simply, just observe its natural behavior.
            # Forcing algorithmic: a bit hacky for a direct test here without refactoring generate_solvable_puzzle.
            # We'll rely on the random chance for now, or you can manually set USE_USER_PUZZLE_CHANCE = 0 before this loop
            # and reset it after if this test block needs to *guarantee* algorithmic.

            puzzle = generate_solvable_puzzle(target_difficulty=diff_test, user_performance_summary=perf_summary)
            print(f"Generated {diff_test.capitalize()} Puzzle Client Data: {puzzle}")
            if puzzle and puzzle.get('puzzle_id') in active_puzzles:
                server_data = active_puzzles[puzzle['puzzle_id']]
                print(f"  Server Data: PuzzleID={server_data['puzzle_id'][:12]}..., ConnType='{server_data['parameters'].get('connection_type')}', ActualDiff={server_data['difficulty']}")
            else:
                print(f"  Failed to generate or store {diff_test} puzzle algorithmically.")
        except Exception as e:
            print(f"Error generating {diff_test} puzzle: {e}")
            logger.error(f"Exception during {diff_test} puzzle generation test:", exc_info=True)

    # Test fetching an approved user puzzle (assuming one exists and is approved)
    if DATABASE_AVAILABLE:
        print("\n--- Testing Fetching Approved User Puzzle ---")
        # Ensure you have an approved puzzle in your DB for this test
        # You might need to run admin_puzzles.html and approve one first
        # Or add one manually via DB Browser with status='approved'
        user_puzzle_data = get_random_approved_user_puzzle()
        if user_puzzle_data:
            print(f"Fetched user puzzle: ID {user_puzzle_data['id']}, Submitter: {user_puzzle_data['submitter_name']}")
            # Now, try to generate a puzzle, hoping it picks this one (increase USE_USER_PUZZLE_CHANCE temporarily if needed for testing)
            print("Attempting to generate a puzzle, which might be the user puzzle...")
            puzzle = generate_solvable_puzzle(target_difficulty="medium") # Or any difficulty
            if puzzle and puzzle.get('puzzle_id') in active_puzzles and puzzle['puzzle_id'].startswith("user_"):
                print(f"  Successfully served user puzzle: {puzzle}")
            elif puzzle:
                print(f"  Generated an algorithmic puzzle instead: {puzzle}")
            else:
                print("  Failed to generate any puzzle after user puzzle test.")
        else:
            print("  No approved user puzzles found to test with.")
    else:
        print("\n--- Skipping User Puzzle Test (Database module not available) ---")


    try:
        print("\n--- Testing Daily Challenge ---")
        daily_data = get_or_generate_daily_challenge()
        print(f"Daily Client Data: {daily_data}")
        if daily_data and daily_data.get('puzzle_id') in active_puzzles:
             print(f"  Daily Server Detail: ActualDiff={active_puzzles[daily_data['puzzle_id']]['difficulty']}")
    except Exception as e:
        print(f"Error with daily challenge: {e}")
        logger.error("Exception during daily challenge test:", exc_info=True)

    cleanup_old_puzzles()
    logger.info("puzzle_logic.py tests completed.")