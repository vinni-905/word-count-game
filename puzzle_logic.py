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
from collections import defaultdict # For grouping words

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
BEST_MODEL_NAME_FROM_TRAINING = "ridge_regression" # CONFIRMED
DIFFICULTY_MODEL_FILENAME = os.path.join(MODEL_DIR, f"wordlinks_{BEST_MODEL_NAME_FROM_TRAINING}_model.pkl")
FEATURE_LIST_FILENAME = os.path.join(MODEL_DIR, "feature_list.pkl")

WORD2VEC_ACTUAL_FILENAME = "glove.6B.100d.kv" # CONFIRMED
WORD2VEC_MODEL_PATH = os.path.join(MODEL_DIR, WORD2VEC_ACTUAL_FILENAME)

ENRICHED_VOCAB_PATH = os.path.join(MODEL_DIR, "enriched_word_data.json")

MAX_GENERATION_ATTEMPTS = 30
WORDS_PER_GROUP = 4
NUM_GROUPS = 4
MAX_PUZZLE_AGE_SECONDS = 3 * 3600
MAX_HINTS = 3

EASY_THRESHOLD = 110
MEDIUM_THRESHOLD = 170

# --- Load Difficulty Prediction ML Model Pipeline ---
model_pipeline = None
ALL_EXPECTED_INPUT_FEATURES = []
try:
    if not BEST_MODEL_NAME_FROM_TRAINING:
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
        if not WORD2VEC_ACTUAL_FILENAME:
            logger.warning(f"WORD2VEC_ACTUAL_FILENAME is not set. W2V hints fallback.")
        elif not os.path.exists(WORD2VEC_MODEL_PATH): logger.warning(f"Word2Vec model '{WORD2VEC_MODEL_PATH}' not found. W2V hints fallback.")
        else:
            logger.info(f"Loading Word2Vec model: {WORD2VEC_MODEL_PATH}...")
            if WORD2VEC_ACTUAL_FILENAME.endswith(".kv"):
                 w2v_model = KeyedVectors.load(WORD2VEC_MODEL_PATH)
            elif WORD2VEC_ACTUAL_FILENAME.endswith(".bin") or WORD2VEC_ACTUAL_FILENAME.endswith(".wordvectors"):
                 w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=WORD2VEC_ACTUAL_FILENAME.endswith(".bin"))
            elif WORD2VEC_ACTUAL_FILENAME.endswith(".model"):
                 w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH).wv
            else:
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
word_categories: Dict[str, List[str]] = {}
if os.path.exists(ENRICHED_VOCAB_PATH):
    try:
        with open(ENRICHED_VOCAB_PATH, 'r', encoding='utf-8') as f: ENRICHED_VOCABULARY = json.load(f)
        logger.info(f"Enriched vocabulary loaded from {ENRICHED_VOCAB_PATH} ({len(ENRICHED_VOCABULARY)} words).")
        temp_word_categories = defaultdict(list)
        for word, data in ENRICHED_VOCABULARY.items():
            category = data.get("category", "unknown_in_enriched")
            temp_word_categories[category].append(word.lower())

        if temp_word_categories and "unknown_in_enriched" not in temp_word_categories and \
           all(len(lst) > 0 for lst in temp_word_categories.values()):
            word_categories = dict(temp_word_categories)
            logger.info(f"Word categories populated from enriched data ({len(word_categories)} categories).")
        else:
            logger.warning("Enriched vocabulary categories incomplete, empty, or contain 'unknown'; falling back to hardcoded.")
            word_categories = {k: [w.lower() for w in v] for k, v in word_categories_hardcoded.items()}
    except Exception as e:
        logger.error(f"Error loading or processing {ENRICHED_VOCAB_PATH}: {e}", exc_info=True)
        word_categories = {k: [w.lower() for w in v] for k, v in word_categories_hardcoded.items()}
else:
    logger.warning(f"'{ENRICHED_VOCAB_PATH}' not found. Using hardcoded categories. Run vocabulary_analyzer.py.")
    word_categories = {k: [w.lower() for w in v] for k, v in word_categories_hardcoded.items()}

connection_types = {
    'same_category': 1, 'begins_with': 2, 'ends_with': 3, 'syllable_count': 4,
    'synonym_groups': 5, 'antonym_groups': 6, 'compound_words': 7,
    'rhyming_words': 7, 'conceptual_relation': 8, 'multiple_rules': 10,
    'letter_pattern': 5, 'anagrams': 9, 'homophones': 6,
    'contains_substring': 4, 'metaphorical_relation': 9,
    'word_length': 3 # ADDED
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
    'word_length': "Same Word Length" # ADDED
}
word_rarity_levels = { 'very_common': 1, 'common': 2, 'somewhat_common': 4, 'uncommon': 6, 'rare': 8, 'very_rare': 10, 'extremely_rare': 12 }
active_puzzles: Dict[str, Dict[str, Any]] = {}

def _get_all_available_words(exclude_words: Optional[Set[str]] = None) -> List[str]:
    if exclude_words is None: exclude_words = set()
    all_words_from_vocab = set()
    if ENRICHED_VOCABULARY:
        all_words_from_vocab.update(word.lower() for word in ENRICHED_VOCABULARY.keys())
    if not all_words_from_vocab and word_categories: # Fallback if enriched is empty
        for cat_words_list in word_categories.values():
            for word in cat_words_list:
                all_words_from_vocab.add(word.lower())
    if not all_words_from_vocab:
        logger.error("_get_all_available_words: No words found.")
        return []
    return list(all_words_from_vocab - {w.lower() for w in exclude_words})

def _get_words_by_target_rarity(category_name: str, target_rarity_name: str, count: int, exclude_list: Set[str]) -> List[str]:
    exclude_list_lower = {w.lower() for w in exclude_list}
    if not ENRICHED_VOCABULARY:
        cat_words = [w.lower() for w in word_categories.get(category_name, []) if w.lower() not in exclude_list_lower]
        return random.sample(cat_words, min(count, len(cat_words))) if len(cat_words) >= count else []
    rarity_to_complexity_map = {
        'very_common': (1, 2), 'common': (3, 4), 'somewhat_common': (5, 6),
        'uncommon': (7, 8), 'rare': (9, 9), 'very_rare': (10, 10), 'extremely_rare': (10,10)
    }
    min_c, max_c = rarity_to_complexity_map.get(target_rarity_name, (1, 10))
    candidate_words = [word.lower() for word, data in ENRICHED_VOCABULARY.items()
                       if data.get("category") == category_name and word.lower() not in exclude_list_lower and
                       min_c <= data.get("complexity_score", 5) <= max_c]
    if len(candidate_words) >= count: return random.sample(candidate_words, count)
    else:
        logger.debug(f"RarityGet: Not enough for cat '{category_name}', rarity '{target_rarity_name}'. Found {len(candidate_words)}. Widening.")
        available_fallback = [word.lower() for word, data in ENRICHED_VOCABULARY.items()
                              if data.get("category") == category_name and word.lower() not in exclude_list_lower]
        if len(available_fallback) >= count: return random.sample(available_fallback, count)
        elif available_fallback : return random.sample(available_fallback, len(available_fallback))
        return []

def _generate_groups_by_category(num_groups_to_gen: int, words_per_group_val: int, used_words_global: Set[str], generation_params: Dict[str, Any]) -> Optional[Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]]:
    target_word_rarity = generation_params.get('word_rarity', 'common')
    all_cat_names = list(word_categories.keys())
    if not all_cat_names:
        logger.error("CatGen: No categories available.")
        return None
    if len(all_cat_names) < num_groups_to_gen:
        logger.warning(f"CatGen: Not enough distinct categories ({len(all_cat_names)}). Using replacement.")
        selected_cat_names = random.choices(all_cat_names, k=num_groups_to_gen)
    else:
        selected_cat_names = random.sample(all_cat_names, num_groups_to_gen)

    solution_groups, all_words_list, descriptions, diff_indices = {}, [], {}, {}
    current_puzzle_used_words = set(used_words_global)
    for i, cat_name in enumerate(selected_cat_names):
        group_id = f"group_{i+1}"
        group_words = _get_words_by_target_rarity(cat_name, target_word_rarity, words_per_group_val, current_puzzle_used_words)
        if len(group_words) < words_per_group_val:
            logger.warning(f"CatGen: Failed for cat '{cat_name}'. Found {len(group_words)} words.")
            return None
        solution_groups[group_id] = sorted(group_words)
        all_words_list.extend(group_words)
        current_puzzle_used_words.update(group_words)
        descriptions[group_id] = connection_descriptions.get('same_category', "Category") + f": {cat_name.replace('_', ' ').capitalize()}"
        diff_indices[group_id] = i
    if len(solution_groups) < num_groups_to_gen : return None
    return solution_groups, all_words_list, descriptions, diff_indices

def _generate_groups_by_starting_letter(num_groups_to_gen: int, words_per_group_val: int, used_words_global: Set[str], generation_params: Dict[str, Any]) -> Optional[Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]]:
    logger.info(f"Attempting to generate {num_groups_to_gen} groups by starting letter.")
    target_word_rarity = generation_params.get('word_rarity', 'common')
    all_available_words_for_this_type = _get_all_available_words(exclude_words=used_words_global)
    if len(all_available_words_for_this_type) < num_groups_to_gen * words_per_group_val:
        logger.warning("StartLetterGen: Not enough available words overall.")
        return None
    words_by_letter = defaultdict(list)
    for word in all_available_words_for_this_type:
        if word: words_by_letter[word[0].lower()].append(word)
    potential_letters = [letter for letter, wl in words_by_letter.items() if len(wl) >= words_per_group_val]
    if len(potential_letters) < num_groups_to_gen:
        logger.warning(f"StartLetterGen: Not enough distinct starting letters. Found {len(potential_letters)}.")
        return None
    selected_letters_for_groups = random.sample(potential_letters, num_groups_to_gen)
    solution_groups, all_words_list, group_descriptions, difficulty_index_map_for_puzzle = {}, [], {}, {}
    current_puzzle_used_words = set(used_words_global)
    for i, letter in enumerate(selected_letters_for_groups):
        group_id = f"group_{i+1}"
        candidate_words_for_letter = []
        if ENRICHED_VOCABULARY and generation_params.get('word_rarity'):
            rarity_map = {'very_common': (1,2), 'common': (3,4), 'somewhat_common': (5,6), 'uncommon': (7,8), 'rare': (9,9), 'very_rare': (10,10), 'extremely_rare': (10,10)}
            min_c, max_c = rarity_map.get(target_word_rarity, (1,10))
            for word in words_by_letter[letter]:
                if word not in current_puzzle_used_words:
                    wd = ENRICHED_VOCABULARY.get(word, {})
                    cs = wd.get("complexity_score", 5)
                    if min_c <= cs <= max_c: candidate_words_for_letter.append(word)
        else:
            candidate_words_for_letter = [w for w in words_by_letter[letter] if w not in current_puzzle_used_words]
        if len(candidate_words_for_letter) < words_per_group_val:
            logger.warning(f"StartLetterGen: Not enough for '{letter}' (rarity '{target_word_rarity}'). Found {len(candidate_words_for_letter)}. Trying any rarity.")
            any_rarity_candidates = [w for w in words_by_letter[letter] if w not in current_puzzle_used_words]
            if len(any_rarity_candidates) >= words_per_group_val: group_words = random.sample(any_rarity_candidates, words_per_group_val)
            else: logger.error(f"StartLetterGen: Still not enough for '{letter}' any rarity. Aborting."); return None
        else: group_words = random.sample(candidate_words_for_letter, words_per_group_val)
        solution_groups[group_id] = sorted(group_words)
        all_words_list.extend(group_words); current_puzzle_used_words.update(group_words)
        group_descriptions[group_id] = f"Words Starting With '{letter.upper()}'"
        difficulty_index_map_for_puzzle[group_id] = i
    if len(solution_groups) == num_groups_to_gen:
        logger.info(f"Successfully generated groups by starting letter.")
        return solution_groups, all_words_list, group_descriptions, difficulty_index_map_for_puzzle
    return None

# NEW FUNCTION: _generate_groups_by_length
def _generate_groups_by_length(num_groups_to_gen: int, words_per_group_val: int, used_words_global: Set[str], generation_params: Dict[str, Any]) -> Optional[Tuple[Dict[str, List[str]], List[str], Dict[str, str], Dict[str, int]]]:
    logger.info(f"Attempting to generate {num_groups_to_gen} groups by word length.")
    target_word_rarity = generation_params.get('word_rarity', 'common')
    all_available_words_for_this_type = _get_all_available_words(exclude_words=used_words_global)

    if len(all_available_words_for_this_type) < num_groups_to_gen * words_per_group_val:
        logger.warning("WordLengthGen: Not enough available words overall to form all groups.")
        return None

    words_by_length = defaultdict(list)
    for word in all_available_words_for_this_type:
        if word: words_by_length[len(word)].append(word)

    potential_lengths = [
        length for length, word_list in words_by_length.items() if len(word_list) >= words_per_group_val
    ]

    if len(potential_lengths) < num_groups_to_gen:
        logger.warning(f"WordLengthGen: Not enough distinct word lengths with sufficient words. Found {len(potential_lengths)} lengths, need {num_groups_to_gen}.")
        return None

    selected_lengths_for_groups = random.sample(potential_lengths, num_groups_to_gen)
    solution_groups, all_words_list, group_descriptions, difficulty_index_map_for_puzzle = {}, [], {}, {}
    current_puzzle_used_words = set(used_words_global)

    for i, length_val in enumerate(selected_lengths_for_groups):
        group_id = f"group_{i+1}"
        candidate_words_for_length = []

        if ENRICHED_VOCABULARY and generation_params.get('word_rarity'):
            rarity_map = {'very_common': (1,2), 'common': (3,4), 'somewhat_common': (5,6), 'uncommon': (7,8), 'rare': (9,9), 'very_rare': (10,10), 'extremely_rare': (10,10)}
            min_c, max_c = rarity_map.get(target_word_rarity, (1,10))
            for word in words_by_length[length_val]:
                if word not in current_puzzle_used_words:
                    word_data = ENRICHED_VOCABULARY.get(word, {})
                    complexity_score = word_data.get("complexity_score", 5)
                    if min_c <= complexity_score <= max_c:
                        candidate_words_for_length.append(word)
        else:
             candidate_words_for_length = [w for w in words_by_length[length_val] if w not in current_puzzle_used_words]
        
        if len(candidate_words_for_length) < words_per_group_val:
            logger.warning(f"WordLengthGen: Not enough words of length {length_val} (rarity '{target_word_rarity}'). Found {len(candidate_words_for_length)}. Trying any rarity.")
            any_rarity_candidates = [w for w in words_by_length[length_val] if w not in current_puzzle_used_words]
            if len(any_rarity_candidates) >= words_per_group_val:
                group_words = random.sample(any_rarity_candidates, words_per_group_val)
            else:
                logger.error(f"WordLengthGen: Still not enough for length {length_val} any rarity ({len(any_rarity_candidates)} found). Aborting.")
                return None
        else:
            group_words = random.sample(candidate_words_for_length, words_per_group_val)

        solution_groups[group_id] = sorted(group_words)
        all_words_list.extend(group_words)
        current_puzzle_used_words.update(group_words)
        group_descriptions[group_id] = f"Words With {length_val} Letters"
        difficulty_index_map_for_puzzle[group_id] = i

    if len(solution_groups) == num_groups_to_gen:
        logger.info(f"Successfully generated {num_groups_to_gen} groups by word length.")
        return solution_groups, all_words_list, group_descriptions, difficulty_index_map_for_puzzle
    else:
        logger.warning("WordLengthGen: Failed to generate all required groups by word length.")
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
    est_time = 20 + score * 4 + (params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP) - (NUM_GROUPS * WORDS_PER_GROUP)) * 0.5
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
            'time_of_day': float(datetime.datetime.now().hour),
            'hints_used': 0.0, 'num_players': 50.0, 'completions': 40.0, 'completion_rate': 0.80, 'attempt_count': 2.0,
            'time_before_first_attempt': 10.0, 'hover_count': float(params.get('num_words', 16) * 1.5),
            'abandonment_rate': 0.20, 'competitiveness_score': 5.0, 'frustration_score': 3.0,
            'satisfaction_score': 7.0, 'learning_value': 5.0, 'engagement_score': 6.0,
            'replayability_score': 4.0, 'avg_attempts_before_success': 1.5
        })
        final_feature_values = [feature_data_dict.get(name, 0.0) for name in ALL_EXPECTED_INPUT_FEATURES]
        if len(final_feature_values) != len(ALL_EXPECTED_INPUT_FEATURES):
            logger.error(f"Feature length mismatch for ML. Expected {len(ALL_EXPECTED_INPUT_FEATURES)}, got {len(final_feature_values)}")
            return get_fallback_prediction(params, actual_selected_words)
        predict_df = pd.DataFrame([final_feature_values], columns=ALL_EXPECTED_INPUT_FEATURES)
        predicted_time = float(model_pipeline.predict(predict_df)[0])
        difficulty = "hard"
        if predicted_time < EASY_THRESHOLD: difficulty = "easy"
        elif predicted_time < MEDIUM_THRESHOLD: difficulty = "medium"
        logger.info(f"ML Prediction: Time={predicted_time:.2f}s, Diff='{difficulty}' for Params: {params} (RarityVal: {word_rarity_for_model_input})")
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

    generation_params, found_matching_params = {}, False
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
            for ct in ['same_category', 'begins_with', 'word_length', 'contains_substring'] # ADDED 'word_length'
            for wr in ['very_common', 'common']
            for sd in [random.uniform(1,2), random.uniform(2,3.5)]] * 5)
    elif actual_search_difficulty == 'medium':
        param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd}
            for ct in ['same_category', 'begins_with', 'word_length', 'syllable_count', 'rhyming_words', 'letter_pattern', 'homophones'] # ADDED 'word_length'
            for wr in ['common', 'somewhat_common', 'uncommon']
            for sd in [random.uniform(2.5,4.5), random.uniform(4,6.5)]] * 4)
    else:
        param_candidates.extend([{'connection_type': ct, 'word_rarity': wr, 'semantic_distance': sd}
            for ct in ['conceptual_relation', 'synonym_groups', 'antonym_groups', 'anagrams', 'metaphorical_relation', 'multiple_rules', 'begins_with', 'word_length'] # ADDED 'word_length'
            for wr in ['uncommon', 'rare', 'very_rare']
            for sd in [random.uniform(5,7), random.uniform(6.5,9)]] * 3)
    if not param_candidates: param_candidates = [{'connection_type': 'same_category', 'word_rarity': 'common', 'semantic_distance': 3.0}] * MAX_GENERATION_ATTEMPTS
    random.shuffle(param_candidates)
    
    predicted_result_for_params = None # Initialize

    for attempt in range(min(MAX_GENERATION_ATTEMPTS, len(param_candidates))):
        current_params = param_candidates[attempt].copy()
        current_params['num_words'] = num_words_total
        if current_params.get('connection_type') not in connection_types or current_params.get('word_rarity') not in word_rarity_levels:
            continue
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
            logger.error("No valid ML predictions or candidates. Using hardcoded fallback.")
            generation_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total, 'semantic_distance': 3.0}
            predicted_result_for_params = get_fallback_prediction(generation_params)

    final_connection_type = generation_params.get('connection_type', 'same_category')
    logger.info(f"--- Finalizing words. Connection: '{final_connection_type}', TargetRarity: '{generation_params.get('word_rarity')}' ---")

    word_selection_data = None
    if final_connection_type == 'same_category':
        word_selection_data = _generate_groups_by_category(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'begins_with':
        word_selection_data = _generate_groups_by_starting_letter(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'word_length': # ADDED
        word_selection_data = _generate_groups_by_length(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'ends_with':
        word_selection_data = _generate_groups_by_ending_letter(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'rhyming_words':
        word_selection_data = _generate_groups_by_rhyme(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    elif final_connection_type == 'anagrams':
        word_selection_data = _generate_groups_by_anagram(NUM_GROUPS, WORDS_PER_GROUP, set(), generation_params)
    else: logger.warning(f"Word selection logic for '{final_connection_type}' is not specifically implemented.")

    if not word_selection_data:
        logger.info(f"Primary word selection failed for '{final_connection_type}'. Fallback groups.")
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
            _generate_fallback_groups(num_words_total, final_connection_type, generation_params)
    else:
        solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = word_selection_data

    if not all_words_for_grid or len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
        logger.critical(f"CRITICAL: Puzzle assembly error. Words: {len(all_words_for_grid if all_words_for_grid else [])}/{num_words_total}, Groups: {len(solution_groups)}/{NUM_GROUPS}. Emergency Fallback.")
        try:
            emergency_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total}
            solution_groups, all_words_for_grid, group_descriptions, difficulty_index_map_for_puzzle = \
                _generate_fallback_groups(num_words_total, "same_category", emergency_params)
            if not all_words_for_grid or len(all_words_for_grid) != num_words_total or len(solution_groups) != NUM_GROUPS:
                 raise ValueError("Emergency fallback also failed.")
        except Exception as e_fall:
            logger.critical(f"Emergency fallback group generation also failed: {e_fall}", exc_info=True)
            solution_groups = {f"dummy_g_{i+1}": [f"word{i*4+j+1}" for j in range(4)] for i in range(4)} # Ensure unique dummy keys
            all_words_for_grid = [word for group in solution_groups.values() for word in group]
            group_descriptions = {k: "Dummy Group (Error)" for k in solution_groups}
            difficulty_index_map_for_puzzle = {k: i for i, k in enumerate(solution_groups)}
            generation_params = {'connection_type': 'same_category', 'word_rarity': 'common', 'num_words': num_words_total}
            predicted_result_for_params = get_fallback_prediction(generation_params, all_words_for_grid) # Ensure this uses the dummy words

    final_puzzle_difficulty_data = predicted_result_for_params # Use prediction that led to these generation_params
    if not (solution_groups and any("dummy_g" in k for k in solution_groups.keys())): # if not the absolute last resort dummy
         # Re-predict with actual words only if not the dummy and if ML is available
        if model_pipeline and ALL_EXPECTED_INPUT_FEATURES:
            final_puzzle_difficulty_data = predict_difficulty_for_params(generation_params, actual_selected_words=all_words_for_grid)
        elif predicted_result_for_params is None: # if somehow it was not set before
            final_puzzle_difficulty_data = get_fallback_prediction(generation_params, all_words_for_grid)
    # else: it's the dummy or ML failed, final_puzzle_difficulty_data is already from predicted_result_for_params

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
    if puzzle_id not in active_puzzles: return {"correct": False, "message": "Invalid or expired puzzle.", "solved_groups": {}}
    data = active_puzzles[puzzle_id]; solution = data["solution"]; descriptions = data["descriptions"]
    diff_idx_map = data.get("parameters", {}).get("difficulty_index_map", {})
    server_solved_groups = data.setdefault("_server_solved_group_keys", set())
    attempt_key = next(iter(user_groups_attempt), None)
    if not attempt_key: return {"correct": False, "message": "No attempt data.", "solved_groups": {}}
    attempt_words = sorted([w.upper() for w in user_groups_attempt[attempt_key]])
    if len(attempt_words) != WORDS_PER_GROUP: return {"correct": False, "message": f"Select {WORDS_PER_GROUP} words.", "solved_groups": {}}
    for gk, correct_words in solution.items():
        if attempt_words == correct_words:
            if gk in server_solved_groups:
                return {"correct": False, "message": f"Already found: {descriptions.get(gk, 'Group')}!", "solved_groups": {}}
            server_solved_groups.add(gk)
            return {"correct": True, "message": f"Correct! Category: {descriptions.get(gk, 'Found')}",
                    "solved_groups": {gk: {"description": descriptions.get(gk, "Unknown"), "difficulty_index": diff_idx_map.get(gk,0)}}}
    return {"correct": False, "message": "Incorrect group.", "solved_groups": {}}

def get_puzzle_hint(puzzle_id: str, solved_group_keys_from_client: Optional[List[str]] = None) -> Dict[str, Any]:
    if solved_group_keys_from_client is None: solved_group_keys_from_client = []
    if puzzle_id not in active_puzzles: return {"hint": None, "message": "Invalid puzzle.", "words": []}
    puzzle_data = active_puzzles[puzzle_id]; actual_solution = puzzle_data.get("solution", {})
    actual_descriptions = puzzle_data.get("descriptions", {}); all_words_on_grid_upper = set(puzzle_data.get("words_on_grid", []))
    server_solved_groups = puzzle_data.get("_server_solved_group_keys", set())
    combined_solved_keys = server_solved_groups.union(set(solved_group_keys_from_client))
    unsolved_groups_details = {k: {"words_upper": v, "words_lower": [w.lower() for w in v], "description": actual_descriptions.get(k, "A Group")}
                       for k, v in actual_solution.items() if k not in combined_solved_keys}
    if not unsolved_groups_details: return {"hint": None, "message": "All groups solved!", "words": []}
    target_group_key = random.choice(list(unsolved_groups_details.keys())); target_group_info = unsolved_groups_details[target_group_key]
    hint_text = f"Hint for '{target_group_info['description']}'."; words_to_highlight_on_grid = []
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
                        found_external_hint_word = sim_word_lower; break
                if found_external_hint_word:
                    hint_text = f"The group '{target_group_info['description']}' might relate to concepts like '{found_external_hint_word.capitalize()}' (not in puzzle)."
                    words_to_highlight_on_grid.append(random.choice(anchor_words_for_w2v).upper())
                else: logger.info("W2V: No suitable distinct external hint word for group '%s'.", target_group_key)
            else: logger.info("W2V: None of the words from group '%s' are in W2V vocab.", target_group_key)
        except Exception as e: logger.error(f"Error in W2V hint gen for group '%s': {e}",target_group_key, exc_info=True)
    if not words_to_highlight_on_grid:
        logger.info("Using basic hint (revealing words from group '%s').", target_group_key)
        num_to_reveal = random.randint(1, min(len(target_group_info["words_upper"]), 2))
        words_to_highlight_on_grid = random.sample(target_group_info["words_upper"], num_to_reveal)
        if words_to_highlight_on_grid: hint_text = f"Hint for '{target_group_info['description']}': Includes {', '.join(words_to_highlight_on_grid)}."
        else: hint_text = f"Try to find words for '{target_group_info['description']}'."
    return {"hint": hint_text, "words": words_to_highlight_on_grid, "message": "Hint provided."}

def cleanup_old_puzzles():
    current_time_dt = datetime.datetime.now(); today_date = datetime.date.today()
    puzzles_to_delete = []
    for pid, data in list(active_puzzles.items()):
        creation_time_dt = data.get("creation_time")
        if not isinstance(creation_time_dt, datetime.datetime):
            if isinstance(creation_time_dt, (float, int)):
                try: creation_time_dt = datetime.datetime.fromtimestamp(creation_time_dt)
                except (OSError, TypeError, ValueError) as ts_err: logger.warning(f"Puzzle {pid} invalid timestamp. Error: {ts_err}. Skip."); continue
            else: logger.warning(f"Puzzle {pid} invalid creation_time type. Skip."); continue
        if pid.startswith("daily_"):
            try:
                puzzle_date_str = pid.replace("daily_", "")
                if datetime.date.fromisoformat(puzzle_date_str) < today_date: puzzles_to_delete.append(pid)
            except ValueError: logger.error(f"Malformed daily ID: {pid}.")
        elif (current_time_dt - creation_time_dt).total_seconds() > MAX_PUZZLE_AGE_SECONDS:
            puzzles_to_delete.append(pid)
    for pid in puzzles_to_delete:
        if pid in active_puzzles: del active_puzzles[pid]; logger.info(f"Cleaned up puzzle: {pid}")

if __name__ == "__main__":
    logger.info("Testing puzzle_logic.py standalone functions...")
    if not model_pipeline: logger.warning("ML Model not loaded for test, expect fallback predictions.")
    if not w2v_model: logger.warning("Word2Vec Model not loaded for test, expect fallback hints.")

    mock_good_perf_medium = {"plays": 5, "win_rate": 0.9, "avg_hints": 0.1}
    mock_struggling_perf_medium = {"plays": 5, "win_rate": 0.2, "avg_hints": 2.5}

    for diff_test in ["easy", "medium", "hard"]:
        try:
            print(f"\n--- Testing Regular Puzzle Generation ({diff_test.capitalize()}) ---")
            perf_summary = mock_good_perf_medium if diff_test == "medium" else (mock_struggling_perf_medium if diff_test == "hard" else None)
            puzzle = generate_solvable_puzzle(target_difficulty=diff_test, user_performance_summary=perf_summary)
            print(f"{diff_test.capitalize()} Puzzle Client Data: {puzzle}")
            if puzzle and puzzle.get('puzzle_id') in active_puzzles:
                server_data = active_puzzles[puzzle['puzzle_id']]
                print(f"  Server Data: SolKeys={list(server_data['solution'].keys())}, ActualDiff={server_data['difficulty']}, PredTime={server_data['predicted_solve_time']:.1f}s, FallbackPred={server_data['is_fallback_prediction']}")
                print(f"  Generation Params: {server_data['parameters']}")
                if server_data['solution']:
                    hint_result = get_puzzle_hint(puzzle['puzzle_id'])
                    print(f"  Hint Test: '{hint_result.get('hint')}', Highlight: {hint_result.get('words')}")
            else: print(f"  Failed to generate/store {diff_test} puzzle.")
        except Exception as e: print(f"Error generating {diff_test} puzzle: {e}", exc_info=True)
    try:
        print("\n--- Testing Daily Challenge ---")
        daily_data = get_or_generate_daily_challenge()
        print(f"Daily Client Data: {daily_data}")
        if daily_data and daily_data.get('puzzle_id') in active_puzzles:
             print(f"  Daily Server Detail: ActualDiff={active_puzzles[daily_data['puzzle_id']]['difficulty']}")
    except Exception as e: print(f"Error with daily challenge: {e}", exc_info=True)
    logger.info("puzzle_logic.py tests completed.")