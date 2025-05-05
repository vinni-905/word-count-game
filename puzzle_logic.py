import random
import string
import uuid
import joblib
import pandas as pd
import numpy as np
import os
import datetime
import warnings
from typing import Dict, Any, Optional, List, Tuple, Set

# --- Constants ---
MODEL_DIR = "model"
# IMPORTANT: Ensure this matches the filename saved by your training script
# Example: If Ridge Regression was best and saved as 'wordlinks_ridge_regression_model.pkl'
BEST_MODEL_NAME_FROM_TRAINING = "ridge_regression" # CHANGE THIS if needed (e.g., "random_forest")
MODEL_FILENAME = os.path.join(MODEL_DIR, f"wordlinks_{BEST_MODEL_NAME_FROM_TRAINING}_model.pkl")
# PREPROCESSOR_FILENAME = os.path.join(MODEL_DIR, "feature_preprocessing.pkl") # No longer loaded separately
FEATURE_LIST_FILENAME = os.path.join(MODEL_DIR, "feature_list.pkl") # Used for input feature reference
MAX_GENERATION_ATTEMPTS = 10 # Max tries to find a puzzle matching difficulty
WORDS_PER_GROUP = 4
NUM_GROUPS = 4

# Suppress specific warnings if needed (e.g., from joblib or sklearn)
# warnings.filterwarnings('ignore', category=UserWarning)

# --- Load Model Pipeline and Feature Info ---
model_pipeline = None
ALL_EXPECTED_INPUT_FEATURES = []

try:
    # Load the *entire pipeline* saved by the training script
    model_pipeline = joblib.load(MODEL_FILENAME)
    # Load the feature info for reference and constructing input
    feature_info = joblib.load(FEATURE_LIST_FILENAME)
    # These are the *original* feature names *before* preprocessing
    TRAINING_NUMERIC_FEATURES = feature_info.get('numeric_features', [])
    TRAINING_CATEGORICAL_FEATURES = feature_info.get('categorical_features', [])
    # Combine to get all features the preprocessor step expects
    ALL_EXPECTED_INPUT_FEATURES = TRAINING_NUMERIC_FEATURES + TRAINING_CATEGORICAL_FEATURES

    print(f"ML Model Pipeline ({MODEL_FILENAME}) loaded successfully.")
    print(f"[DEBUG] Expected input features for preprocessor: {ALL_EXPECTED_INPUT_FEATURES}")
    print(f"[DEBUG] Number of expected input features: {len(ALL_EXPECTED_INPUT_FEATURES)}")

except FileNotFoundError:
    print(f"[ERROR] Model pipeline ({MODEL_FILENAME}) or feature list ({FEATURE_LIST_FILENAME}) not found in '{MODEL_DIR}'. ML prediction disabled.")
    model_pipeline = None
    ALL_EXPECTED_INPUT_FEATURES = []
except Exception as e:
    print(f"[ERROR] Error loading ML model pipeline: {e}. ML prediction disabled.")
    model_pipeline = None
    ALL_EXPECTED_INPUT_FEATURES = []


# --- Data Definitions (Keep or adapt from original script) ---
word_categories = {
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'brown', 'gray', 'teal', 'maroon', 'indigo', 'violet'],
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'snake', 'eagle'],
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach', 'kiwi', 'watermelon', 'cherry', 'pear', 'blueberry', 'raspberry', 'lemon'],
    'vehicles': ['car', 'bus', 'train', 'airplane', 'bicycle', 'motorcycle', 'truck', 'boat', 'helicopter', 'submarine', 'scooter', 'tractor', 'van', 'ambulance', 'taxi'],
    'sports': ['soccer', 'basketball', 'tennis', 'golf', 'swimming', 'baseball', 'volleyball', 'football', 'hockey', 'rugby', 'cricket', 'boxing', 'skiing', 'climbing', 'cycling'],
    'countries': ['usa', 'canada', 'japan', 'brazil', 'france', 'germany', 'india', 'australia', 'china', 'mexico', 'italy', 'spain', 'russia', 'egypt', 'kenya'],
    'professions': ['doctor', 'teacher', 'engineer', 'lawyer', 'chef', 'artist', 'scientist', 'writer', 'nurse', 'programmer', 'architect', 'accountant', 'journalist', 'pilot', 'firefighter'],
    'instruments': ['piano', 'guitar', 'violin', 'drums', 'flute', 'trumpet', 'saxophone', 'clarinet', 'harp', 'cello', 'banjo', 'harmonica', 'trombone', 'accordion', 'ukulele'],
    'weather': ['sunny', 'rainy', 'cloudy', 'snowy', 'windy', 'stormy', 'foggy', 'humid', 'dry', 'freezing', 'tropical', 'mild', 'chilly', 'breezy', 'thunderous'],
    'emotions': ['happy', 'sad', 'angry', 'excited', 'fearful', 'surprised', 'disgusted', 'anxious', 'calm', 'bored', 'content', 'proud', 'envious', 'jealous', 'grateful'],
    'furniture': ['table', 'chair', 'couch', 'dresser', 'bed', 'desk', 'bookcase', 'stool', 'cabinet', 'wardrobe', 'bench', 'ottoman', 'armchair', 'nightstand', 'futon'],
    'body_parts': ['head', 'arm', 'leg', 'foot', 'hand', 'eye', 'ear', 'nose', 'mouth', 'finger', 'elbow', 'knee', 'shoulder', 'neck', 'ankle'],
    'foods': ['pizza', 'pasta', 'burger', 'salad', 'soup', 'sandwich', 'taco', 'rice', 'bread', 'cheese', 'fish', 'steak', 'cake', 'cookie', 'chocolate'],
    'clothing': ['shirt', 'pants', 'dress', 'skirt', 'jacket', 'coat', 'hat', 'shoe', 'sock', 'scarf', 'glove', 'belt', 'tie', 'sweater', 'shorts'],
    'elements': ['fire', 'water', 'earth', 'air', 'metal', 'wood', 'electricity', 'light', 'shadow', 'ice', 'steam', 'plasma', 'space', 'void', 'energy']
}

# Use complexity values from training script
connection_types = {
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

# Use rarity levels from training script
word_rarity_levels = {
    'very_common': 1, 'common': 2, 'somewhat_common': 4, 'uncommon': 6,
    'rare': 8, 'very_rare': 10, 'extremely_rare': 12
}

# In-memory storage for active puzzles and their solutions
active_puzzles: Dict[str, Dict[str, Any]] = {}

# --- ML Prediction Helper ---

def get_fallback_prediction(params: Dict[str, Any]) -> Dict[str, Any]:
    """Provides a default prediction structure when ML fails."""
    print("[WARN] Using fallback difficulty estimation.")
    complexity = connection_types.get(params.get('connection_type', 'same_category'), 1)
    rarity = word_rarity_levels.get(params.get('word_rarity', 'common'), 2)
    num_words = params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP)
    # Use thresholds from training script's prediction function
    est_time = 20 + (complexity * 4) + (rarity * 3) + ((num_words - 4) * 5) # Rough estimate based on training generation logic
    if est_time < 40: difficulty = "easy"
    elif est_time < 80: difficulty = "medium"
    else: difficulty = "hard"
    return {'predicted_solve_time': round(est_time, 2), 'difficulty': difficulty}

def predict_difficulty_for_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the loaded ML model pipeline to predict solve time and difficulty.
    Constructs the input DataFrame to match the training data structure.
    """
    if not model_pipeline or not ALL_EXPECTED_INPUT_FEATURES:
        return get_fallback_prediction(params)

    # --- Map generation parameters to feature values ---
    connection_complexity = connection_types.get(params['connection_type'], 5)
    word_rarity_value = word_rarity_levels.get(params['word_rarity'], 5)
    current_hour = datetime.datetime.now().hour
    num_words = params.get('num_words', NUM_GROUPS * WORDS_PER_GROUP)

    # --- Create DataFrame matching the TRAINING 'X' structure ---
    feature_data = {
        # --- Columns directly controllable or derivable ---
        'num_words': float(num_words),
        'connection_complexity': float(connection_complexity),
        'word_rarity_value': float(word_rarity_value),
        'semantic_distance': float(params.get('semantic_distance', 5.0)),
        'time_of_day': current_hour,
        'hints_used': float(params.get('hints_used', 0)),

        # --- Columns needing default values (use same as training example) ---
        'num_players': 50.0,
        'completions': 40.0,
        'completion_rate': 80.0,
        'attempt_count': 2.0,
        'time_before_first_attempt': 10.0,
        'hover_count': float(num_words * 2),
        'abandonment_rate': 20.0,
        'competitiveness_score': 5.0,
        'frustration_score': 5.0,
        'satisfaction_score': 6.0,
        'learning_value': 5.0,
        'engagement_score': 5.5,
        'replayability_score': 5.0,
        'avg_attempts_before_success': 1.5
    }

    # Create DataFrame
    predict_df = pd.DataFrame([feature_data])
    print(f"[DEBUG] Initial predict_df columns: {list(predict_df.columns)}")

    # --- Ensure DataFrame has EXACT columns in CORRECT order ---
    try:
        predict_df_ordered = predict_df[ALL_EXPECTED_INPUT_FEATURES]
        print(f"[DEBUG] Columns in predict_df_ordered (sent to pipeline): {list(predict_df_ordered.columns)}")
        print(f"[DEBUG] Shape of predict_df_ordered (sent to pipeline): {predict_df_ordered.shape}")
    except KeyError as e:
        print(f"[ERROR] Missing expected columns for prediction: {e}")
        return get_fallback_prediction(params)
    except Exception as e:
        print(f"[ERROR] Unexpected error preparing DataFrame for prediction: {e}")
        return get_fallback_prediction(params)

    # --- Make Prediction using the full pipeline ---
    try:
        predicted_time = model_pipeline.predict(predict_df_ordered)[0]
    except ValueError as e:
         print(f"[ERROR] ValueError during pipeline prediction: {e}")
         print(f"Data sent to pipeline:\n{predict_df_ordered.iloc[0].to_dict()}")
         return get_fallback_prediction(params)
    except Exception as e:
        print(f"[ERROR] Unexpected error during pipeline prediction: {e}")
        print(f"Data sent to pipeline:\n{predict_df_ordered.iloc[0].to_dict()}")
        return get_fallback_prediction(params)

    # --- Determine difficulty category (using training thresholds) ---
    if predicted_time < 40: difficulty = "easy"
    elif predicted_time < 80: difficulty = "medium"
    else: difficulty = "hard"

    return {
        'predicted_solve_time': round(predicted_time, 2),
        'difficulty': difficulty
    }


# --- Puzzle Generation Logic ---

def _get_all_available_words() -> Set[str]:
    """Helper to get a unique set of all words from categories."""
    return set(word.lower() for cat_words in word_categories.values() for word in cat_words)

def _generate_fallback_groups(num_words_needed: int, words_per_group: int, connection_type_desc: str) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str]]:
    """Generates random groups as a fallback."""
    solution_groups = {}
    all_words_list = []
    group_descriptions = {}
    all_available_words = list(_get_all_available_words())

    if len(all_available_words) < num_words_needed:
         raise ValueError(f"Not enough unique words ({len(all_available_words)}) available for fallback.")

    print(f"[WARN] Using random categories as fallback for '{connection_type_desc}'.")
    temp_all_words = random.sample(all_available_words, num_words_needed)
    random.shuffle(temp_all_words)
    group_num = 1
    for i in range(0, num_words_needed, words_per_group):
         group_words = temp_all_words[i:i+words_per_group]
         if len(group_words) == words_per_group:
             group_id = f"group_{group_num}"
             solution_groups[group_id] = sorted(group_words)
             all_words_list.extend(group_words)
             # Use the *intended* description from the generation parameters
             group_descriptions[group_id] = connection_descriptions.get(connection_type_desc, "Related Words") + f" (Group {group_num})" # Generic ID added
             group_num += 1
    return solution_groups, all_words_list, group_descriptions


def generate_solvable_puzzle(target_difficulty: str, words_per_group: int = WORDS_PER_GROUP) -> Dict[str, Any]:
    """
    Generates a puzzle, attempts to match target difficulty using ML prediction,
    and selects words based on the connection type (with fallbacks).
    """
    num_groups = NUM_GROUPS
    num_words = num_groups * words_per_group
    puzzle_id = str(uuid.uuid4())
    generation_params = {}
    predicted_result = {}
    last_successful_params = None
    last_successful_pred = {}

    print(f"\n--- Attempting to generate puzzle for difficulty: {target_difficulty} ---")

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        # 1. Choose Candidate Parameters (using ranges from training script)
        if target_difficulty == 'easy':
            conn_type_name = random.choice(['same_category', 'begins_with', 'ends_with', 'contains_substring']) # Expanded easy types
            word_rarity_name = random.choice(['very_common', 'common'])
            semantic_distance = random.uniform(1, 4)
        elif target_difficulty == 'medium':
            conn_type_name = random.choice(['syllable_count', 'letter_pattern', 'rhyming_words', 'synonym_groups', 'homophones', 'antonym_groups', 'compound_words']) # Expanded medium types
            word_rarity_name = random.choice(['common', 'somewhat_common', 'uncommon'])
            semantic_distance = random.uniform(3, 7)
        else: # hard
            conn_type_name = random.choice(['conceptual_relation', 'anagrams', 'metaphorical_relation', 'multiple_rules']) # Hard types
            word_rarity_name = random.choice(['uncommon', 'rare', 'very_rare', 'extremely_rare'])
            semantic_distance = random.uniform(6, 10)

        if conn_type_name not in connection_types:
             print(f"[WARN] Attempt {attempt+1}: Chosen connection type '{conn_type_name}' not recognized, skipping.")
             continue

        generation_params = {
            'num_words': num_words,
            'connection_type': conn_type_name,
            'word_rarity': word_rarity_name,
            'semantic_distance': semantic_distance,
        }

        # 2. Predict Difficulty
        try:
            predicted_result = predict_difficulty_for_params(generation_params)
            if 'predicted_solve_time' not in predicted_result: # Check for fallback indicator if any
                 print(f"[WARN] Attempt {attempt+1}: Prediction failed for params: {generation_params}. Skipping.")
                 continue
            print(f"Attempt {attempt+1}: Params: {conn_type_name}/{word_rarity_name} (SD: {semantic_distance:.2f}), Predicted: {predicted_result['difficulty']} ({predicted_result['predicted_solve_time']}s)")
            last_successful_params = generation_params.copy()
            last_successful_pred = predicted_result.copy()

        except Exception as e:
            print(f"[ERROR] Attempt {attempt+1}: Error during prediction for params {generation_params}: {e}")
            continue

        # 3. Check if prediction matches target
        if predicted_result.get('difficulty') == target_difficulty:
            print(f"--> Found suitable parameters matching '{target_difficulty}' on attempt {attempt+1}.")
            break
    else:
        print(f"[WARN] Could not generate parameters matching '{target_difficulty}' after {MAX_GENERATION_ATTEMPTS} attempts.")
        if last_successful_params:
             print("--> Using parameters from the last successful prediction attempt.")
             generation_params = last_successful_params
             predicted_result = last_successful_pred
        else:
             print("[ERROR] No successful predictions were made. Cannot generate puzzle.")
             # Return an error structure or raise an exception for the API layer
             # return {"error": "Failed to generate puzzle parameters."} # Option 1
             raise ValueError("Failed to generate suitable puzzle parameters after prediction attempts.") # Option 2

    # --- Word Selection/Puzzle Assembly ---
    final_connection_type = generation_params.get('connection_type', 'same_category')
    solution_groups = {}
    all_words = []
    group_descriptions = {}
    generation_successful = False # Flag to track if specific logic succeeded

    print(f"--- Finalizing puzzle generation with type: {final_connection_type} ---")

    try:
        if final_connection_type == 'same_category':
            print(f"[INFO] Attempting to generate words for '{final_connection_type}'...")
            if len(word_categories) < num_groups:
                raise ValueError("Not enough word categories defined.")
            selected_categories = random.sample(list(word_categories.keys()), num_groups)
            for i, category_name in enumerate(selected_categories):
                 available_words = word_categories.get(category_name, [])
                 if len(available_words) < words_per_group:
                      print(f"[WARN] Category '{category_name}' has < {words_per_group} words. Skipping group.")
                      continue # Skip group if not enough words
                 group_words = random.sample(available_words, words_per_group)
                 group_id = f"group_{i+1}"
                 solution_groups[group_id] = sorted([w.lower() for w in group_words])
                 all_words.extend(solution_groups[group_id])
                 group_descriptions[group_id] = connection_descriptions.get(final_connection_type) + f": {category_name.capitalize()}"
            if len(solution_groups) == num_groups: generation_successful = True

        elif final_connection_type == 'ends_with':
            print(f"[INFO] Attempting to generate words for '{final_connection_type}'...")
            all_available = list(_get_all_available_words())
            words_by_ending = {}
            target_suffix_length = 1
            for word in all_available:
                if len(word) >= target_suffix_length:
                    ending = word[-target_suffix_length:]
                    if ending not in words_by_ending: words_by_ending[ending] = []
                    words_by_ending[ending].append(word)

            valid_endings = [e for e, w in words_by_ending.items() if len(w) >= words_per_group]
            if len(valid_endings) >= num_groups:
                selected_endings = random.sample(valid_endings, num_groups)
                used_words = set()
                for i, ending in enumerate(selected_endings):
                    possible_words = [w for w in words_by_ending[ending] if w not in used_words]
                    if len(possible_words) >= words_per_group:
                         group_words = random.sample(possible_words, words_per_group)
                         group_id = f"group_{i+1}"
                         solution_groups[group_id] = sorted(group_words)
                         all_words.extend(group_words)
                         used_words.update(group_words)
                         group_descriptions[group_id] = connection_descriptions.get(final_connection_type) + f" (Ending: '{ending}')"
                if len(solution_groups) == num_groups: generation_successful = True


        elif final_connection_type == 'begins_with':
            print(f"[INFO] Attempting to generate words for '{final_connection_type}'...")
            all_available = list(_get_all_available_words())
            words_by_beginning = {}
            target_prefix_length = 1
            for word in all_available:
                if len(word) >= target_prefix_length:
                    beginning = word[:target_prefix_length]
                    if beginning not in words_by_beginning: words_by_beginning[beginning] = []
                    words_by_beginning[beginning].append(word)

            valid_beginnings = [b for b, w in words_by_beginning.items() if len(w) >= words_per_group]
            if len(valid_beginnings) >= num_groups:
                selected_beginnings = random.sample(valid_beginnings, num_groups)
                used_words = set()
                for i, beginning in enumerate(selected_beginnings):
                    possible_words = [w for w in words_by_beginning[beginning] if w not in used_words]
                    if len(possible_words) >= words_per_group:
                         group_words = random.sample(possible_words, words_per_group)
                         group_id = f"group_{i+1}"
                         solution_groups[group_id] = sorted(group_words)
                         all_words.extend(group_words)
                         used_words.update(group_words)
                         group_descriptions[group_id] = connection_descriptions.get(final_connection_type) + f" (Starting: '{beginning}')"
                if len(solution_groups) == num_groups: generation_successful = True

        # Add elif blocks here for 'rhyming_words', 'anagrams', etc.

        # --- Fallback if specific logic failed or wasn't implemented ---
        if not generation_successful:
             solution_groups, all_words, group_descriptions = _generate_fallback_groups(num_words, words_per_group, final_connection_type)
             # Ensure fallback produced the required number of groups
             if len(solution_groups) != num_groups:
                  raise ValueError("Fallback generation failed to produce the required number of groups.")

    except Exception as e:
         print(f"[ERROR] Exception during word selection for type '{final_connection_type}': {e}. Attempting fallback.")
         try:
             solution_groups, all_words, group_descriptions = _generate_fallback_groups(num_words, words_per_group, final_connection_type)
             if len(solution_groups) != num_groups:
                 raise ValueError("Fallback generation failed after an exception.")
         except Exception as fallback_e:
             print(f"[ERROR] Fallback generation also failed: {fallback_e}")
             raise ValueError("Failed to generate puzzle words using both specific logic and fallback.") from fallback_e


    # Final check - Ensure we have the correct number of words overall
    if len(all_words) != num_words:
        print(f"[ERROR] Final word count mismatch. Expected {num_words}, got {len(all_words)}. Puzzle invalid.")
        raise ValueError("Generated puzzle has incorrect number of words.")

    random.shuffle(all_words) # Shuffle for the player

    # Store the solution details
    active_puzzles[puzzle_id] = {
        "solution": solution_groups, # Already sorted lists of lowercase words
        "descriptions": group_descriptions,
        "difficulty": predicted_result.get('difficulty', target_difficulty),
        "predicted_solve_time": predicted_result.get('predicted_solve_time', -1),
        "parameters": generation_params,
        "creation_time": datetime.datetime.now() # For potential cleanup later
    }
    print(f"--- Puzzle {puzzle_id} generated successfully. ---")

    # Return data for the frontend
    return {
        "puzzle_id": puzzle_id,
        "words": all_words, # Shuffled list for display
        "num_words": len(all_words),
        "difficulty": predicted_result.get('difficulty', target_difficulty),
        "predicted_solve_time": predicted_result.get('predicted_solve_time', -1)
    }

# --- Answer Checking ---
def check_puzzle_answer(puzzle_id: str, user_groups: Dict[str, List[str]]) -> Dict[str, Any]:
    """Checks the user's submitted groups against the stored solution."""
    if puzzle_id not in active_puzzles:
        return {"correct": False, "message": "Invalid or expired puzzle ID.", "solved_groups": {}}

    puzzle_info = active_puzzles[puzzle_id]
    correct_solution = puzzle_info["solution"] # Stored as sorted lists of lowercase words
    group_descriptions = puzzle_info["descriptions"]
    num_expected_groups = len(correct_solution)

    normalized_user_groups = {}
    for group_key, words in user_groups.items():
        if isinstance(words, list) and len(words) > 0:
             # Normalize user input: lowercase and sort
             normalized_user_groups[group_key] = tuple(sorted([w.lower() for w in words]))
        else:
             print(f"[WARN] Invalid group format received for puzzle {puzzle_id}: {group_key} -> {words}")
             # Skip invalid groups instead of failing

    # Correct solution is already stored as sorted lists, convert to tuples for set comparison
    correct_solution_tuples = {key: tuple(words) for key, words in correct_solution.items()}
    correct_word_sets = set(correct_solution_tuples.values())

    submitted_word_sets = set(normalized_user_groups.values())

    # Find correctly identified groups by comparing tuples
    correctly_identified_groups = {}
    matched_solution_keys = set()
    for solution_key, solution_tuple in correct_solution_tuples.items():
         if solution_tuple in submitted_word_sets and solution_key not in matched_solution_keys:
             correctly_identified_groups[solution_key] = group_descriptions.get(solution_key, "Correct Group")
             matched_solution_keys.add(solution_key)

    num_correct_user_submitted = len(correctly_identified_groups)
    is_full_solve = (submitted_word_sets == correct_word_sets) and (len(normalized_user_groups) == num_expected_groups)

    if is_full_solve:
        solution_details = active_puzzles.pop(puzzle_id) # Remove solved puzzle
        return {
            "correct": True,
            "message": "Congratulations! You solved the puzzle!",
            "solved_groups": solution_details["descriptions"] # Return all descriptions
        }
    else:
        message = f"You found {num_correct_user_submitted} out of {num_expected_groups} groups."
        if num_correct_user_submitted > 0:
            message += " Keep trying!"
        else:
            message = "Incorrect guess. Try again!"

        return {
            "correct": False,
            "message": message,
            "solved_groups": correctly_identified_groups # Descriptions only for correct groups found so far
        }

# --- Hint Logic ---
def get_puzzle_hint(puzzle_id: str, solved_group_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Provides a hint (description) for an unsolved group."""
    if solved_group_keys is None:
        solved_group_keys = []

    if puzzle_id not in active_puzzles:
        return {"hint": None, "message": "Invalid or expired puzzle ID."}

    puzzle_info = active_puzzles[puzzle_id]
    solution = puzzle_info.get("solution", {})
    descriptions = puzzle_info.get("descriptions", {})

    if not solution or not descriptions:
         return {"hint": None, "message": "Puzzle data incomplete."}

    all_group_keys = list(solution.keys())
    valid_solved_keys = [key for key in solved_group_keys if key in all_group_keys]
    unsolved_group_keys = [key for key in all_group_keys if key not in valid_solved_keys]

    if not unsolved_group_keys:
        return {"hint": None, "message": "No more hints available - all groups solved or identified!"}

    # Provide hint for a randomly chosen unsolved group
    hint_group_key = random.choice(unsolved_group_keys)
    hint_description = descriptions.get(hint_group_key, "Related Words")

    # Extract the core category part if description is formatted like "Type: Category"
    if ":" in hint_description:
        hint_category = hint_description.split(":")[-1].strip()
    else:
        hint_category = hint_description

    # Return the *full* description as the hint text for clarity
    hint_text = f"Hint: One remaining group is '{hint_description}'."

    return {"hint": hint_text, "message": "Hint provided."}


# --- Cleanup Function (Optional) ---
def cleanup_old_puzzles(max_age_seconds=3600):
    """Removes puzzles older than max_age_seconds from memory."""
    now = datetime.datetime.now()
    puzzles_to_remove = [
        pid for pid, data in active_puzzles.items()
        if (now - data.get("creation_time", now)).total_seconds() > max_age_seconds
    ]
    for pid in puzzles_to_remove:
        print(f"[INFO] Cleaning up old puzzle: {pid}")
        active_puzzles.pop(pid, None)

# You might want to run cleanup periodically, e.g., using a background task scheduler
# if this were a longer-running application.