import json
import os
import logging
import time # For potential rate limiting if using APIs heavily
from typing import Dict, List, Any, Set, Optional, Tuple # Ensure Tuple and Optional are here

# Attempt to import requests, pyphen. They are optional for basic functionality.
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None # Define to prevent runtime errors if called when not available

_pyphen_module = None
_PyphenClass = None
PYPHEN_AVAILABLE = False # Default to False
try:
    import pyphen as pyphen_module_imported 
    _pyphen_module = pyphen_module_imported
    if hasattr(_pyphen_module, 'Pyphen'): # Check if Pyphen class exists
        _PyphenClass = _pyphen_module.Pyphen 
        PYPHEN_AVAILABLE = True # Only set to True if class can be referenced
except ImportError:
    pass # PYPHEN_AVAILABLE remains False

# --- Initialize Logger ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

if PYPHEN_AVAILABLE and _PyphenClass:
    logger.info("Pyphen module and class available.")
elif PYPHEN_AVAILABLE and not _PyphenClass:
    logger.warning("Pyphen module imported, but Pyphen class not found within it. Syllable counting may fail.")
else:
    logger.warning("Pyphen library not found or importable. Syllable counting will be skipped.")

if REQUESTS_AVAILABLE:
    logger.info("Requests library available for API calls.")
else:
    logger.warning("Requests library not found. Definition fetching via API will be skipped.")


# --- Configuration ---
WORD_CATEGORIES_FOR_ANALYSIS: Dict[str, List[str]] = {
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'brown', 'gray', 'teal', 'maroon', 'indigo', 'violet', 'cyan', 'magenta', 'lime', 'olive', 'gold', 'silver', 'bronze', 'ivory', 'beige', 'ubiquitous', 'serendipity'],
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'snake', 'eagle', 'shark', 'whale', 'dolphin', 'penguin', 'owl', 'bat', 'bee', 'ant', 'aardvark', 'capybara'],
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach', 'kiwi', 'watermelon', 'cherry', 'pear', 'blueberry', 'raspberry', 'lemon', 'lime', 'plum', 'apricot', 'fig', 'date', 'guava'],
    'kitchen_items': ['knife', 'fork', 'spoon', 'plate', 'bowl', 'cup', 'pan', 'pot', 'oven', 'microwave', 'blender', 'toaster', 'kettle', 'grater', 'whisk', 'ladle', 'spatula', 'colander']
    # !!! ADD ALL YOUR CATEGORIES FROM puzzle_logic.py HERE for full analysis !!!
}

MODEL_DIR = "model" 
OUTPUT_FILENAME = os.path.join(MODEL_DIR, "enriched_word_data.json")

# --- API Function (Definition Fetching) ---
def get_definition_from_api(word: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetches definition and an example sentence from dictionaryapi.dev"""
    if not REQUESTS_AVAILABLE or not requests: # Check if requests module is usable
        logger.debug(f"Requests library not available, skipping API call for '{word}'.")
        return None, None
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=5)
        response.raise_for_status() 
        data = response.json()
        
        if isinstance(data, list) and data:
            meanings = data[0].get("meanings", [])
            if meanings and isinstance(meanings, list) and meanings[0].get("definitions"):
                first_definition_obj = meanings[0]["definitions"][0]
                definition = first_definition_obj.get("definition")
                example = first_definition_obj.get("example") 

                if definition and len(definition) > 200: definition = definition[:197] + "..."
                if example and len(example) > 150: example = example[:147] + "..."
                return definition, example
    except requests.exceptions.RequestException as e:
        logger.warning(f"API request failed for word '{word}': {e}")
    except (KeyError, IndexError, TypeError) as e: 
        logger.warning(f"Error parsing API response for word '{word}': {e}")
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON response for word '{word}'. Error: {e}")
    return None, None

# --- Complexity Scoring Logic ---
def get_word_complexity_score(word: str, pyphen_dic_instance: Optional[Any] = None) -> int: # pyphen_dic_instance is an instance of _PyphenClass or None
    word_lower = word.lower()
    complexity = 0

    if len(word_lower) > 9: complexity += 3
    elif len(word_lower) > 6: complexity += 2
    elif len(word_lower) > 4: complexity += 1

    if PYPHEN_AVAILABLE and pyphen_dic_instance: # Check if the instance was successfully created
        try:
            syllables = len(pyphen_dic_instance.inserted(word_lower).split('-'))
            if syllables > 3: complexity += 4
            elif syllables > 2: complexity += 2
            elif syllables > 1: complexity += 1
        except Exception as e:
            logger.debug(f"Pyphen syllable count error for '{word_lower}': {e}")
    
    very_common_words = {"the", "a", "is", "to", "and", "of", "it", "in", "on", "for", "be", "red", "blue", "dog", "cat"}
    if word_lower in very_common_words and len(word_lower) <= 4:
        complexity = max(0, complexity - 2) 

    return min(10, max(1, complexity))

# --- Main Analysis Function ---
def analyze_vocabulary(word_categories_data: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    logger.info("Starting vocabulary analysis...")
    enriched_vocabulary: Dict[str, Dict[str, Any]] = {}
    unique_words: Set[str] = set()

    for category, words in word_categories_data.items():
        for word in words:
            unique_words.add(word.lower())

    pyphen_dic_object: Optional[Any] = None # Use Optional[Any] or Optional[_PyphenClass]
    if PYPHEN_AVAILABLE and _PyphenClass: # Check if the class was successfully obtained
        try:
            pyphen_dic_object = _PyphenClass(lang='en_US') 
            logger.info("Pyphen dictionary initialized for syllable counting.")
        except Exception as e: 
            logger.warning(f"Could not initialize Pyphen object: {e}. Syllable counting will be skipped.")
    elif PYPHEN_AVAILABLE:
        logger.warning("Pyphen module was imported, but Pyphen class reference (_PyphenClass) seems missing. Check import logic.")
    else:
        logger.warning("Pyphen library not available. Syllable counting will be skipped.")

    logger.info(f"Analyzing {len(unique_words)} unique words...")
    for i, word_lower in enumerate(list(unique_words)):
        original_category = "unknown"
        for category, words_in_cat in word_categories_data.items():
            if word_lower in [w.lower() for w in words_in_cat]:
                original_category = category
                break
        
        complexity_score = get_word_complexity_score(word_lower, pyphen_dic_instance=pyphen_dic_object) # Pass the initialized object
        
        definition, example = None, None # Initialize
        if REQUESTS_AVAILABLE: # Only call if requests library is present
            definition, example = get_definition_from_api(word_lower)
            if (i + 1) % 10 == 0: # Log progress less frequently
                logger.info(f"Processed {i+1}/{len(unique_words)} words for definitions (API calls may be slow)...")
                time.sleep(0.1) # Small delay to be nice to the API, adjust as needed
        elif (i + 1) % 100 == 0: # Log progress for complexity scoring if API is off
             logger.info(f"Processed {i+1}/{len(unique_words)} words for complexity (API for definitions disabled).")


        enriched_vocabulary[word_lower] = {
            "category": original_category,
            "complexity_score": complexity_score,
            "definition": definition if definition else "", 
            "example_sentence": example if example else ""  
        }
    logger.info("Vocabulary analysis complete.")
    return enriched_vocabulary

if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logger.info(f"Created directory: {MODEL_DIR}")

    logger.info(f"Using WORD_CATEGORIES_FOR_ANALYSIS with {len(WORD_CATEGORIES_FOR_ANALYSIS)} categories.")
    enriched_data = analyze_vocabulary(WORD_CATEGORIES_FOR_ANALYSIS)

    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=4, sort_keys=True)
        logger.info(f"Enriched vocabulary successfully saved to: {OUTPUT_FILENAME}")
        
        logger.info("\n--- Example Enriched Words (first 5 + specific examples) ---")
        count = 0
        specific_check = ["ubiquitous", "serendipity", "aardvark", "red", "apple"]
        for word, data in enriched_data.items():
            if count < 5 or word in specific_check:
                if word in enriched_data: 
                    logger.info(f"'{word}': {enriched_data[word]}")
            count += 1
            printed_specific_count = sum(1 for s_word in specific_check if s_word in enriched_data and count > 5) 
            if count > 5 and printed_specific_count >= len([s for s in specific_check if s in enriched_data]):
                 if count > 15: break 
                 
    except IOError as e:
        logger.error(f"Error writing enriched vocabulary to JSON file: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during saving: {e}", exc_info=True)