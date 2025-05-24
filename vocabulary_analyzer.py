# vocabulary_analyzer.py
import json
import os
import logging
from typing import Dict, List, Any, Set, Optional 

# Optional: Import NLP libraries if you use them for complexity scoring
# Ensure these are installed in your environment: pip install pyphen
_pyphen_module = None # Store the module itself
_PyphenClass = None   # Store the class Pyphen from the module

try:
    import pyphen as pyphen_module_imported # Import the module with an alias
    _pyphen_module = pyphen_module_imported
    _PyphenClass = _pyphen_module.Pyphen # Get the class from the imported module
    PYPHEN_AVAILABLE = True
    logger = logging.getLogger(__name__) # Define logger after potential pyphen import
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')
    logger.info("Pyphen module imported.")
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')
    logger.warning("Pyphen library not found. Syllable counting will be skipped.")
    PYPHEN_AVAILABLE = False


# --- Constants ---
WORD_CATEGORIES_FOR_ANALYSIS: Dict[str, List[str]] = {
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'brown', 'gray', 'teal', 'maroon', 'indigo', 'violet', 'cyan', 'magenta', 'lime', 'olive', 'gold', 'silver', 'bronze', 'ivory', 'beige', 'ubiquitous', 'serendipity'],
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'snake', 'eagle', 'shark', 'whale', 'dolphin', 'penguin', 'owl', 'bat', 'bee', 'ant', 'aardvark', 'capybara'],
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach', 'kiwi', 'watermelon', 'cherry', 'pear', 'blueberry', 'raspberry', 'lemon', 'lime', 'plum', 'apricot', 'fig', 'date', 'guava'],
    'kitchen_items': ['knife', 'fork', 'spoon', 'plate', 'bowl', 'cup', 'pan', 'pot', 'oven', 'microwave', 'blender', 'toaster', 'kettle', 'grater', 'whisk', 'ladle', 'spatula', 'colander']
    # !!! ADD ALL YOUR CATEGORIES FROM puzzle_logic.py HERE !!!
}

MODEL_DIR = "model" 
OUTPUT_FILENAME = os.path.join(MODEL_DIR, "enriched_word_data.json")

# --- Complexity Scoring Logic ---
# Corrected type hint for pyphen_dic: it's an instance of the Pyphen class
def get_word_complexity_score(word: str, pyphen_dic: Optional[Any] = None) -> int: # Use Any for pyphen_dic if class isn't globally typed
    # Or, more precisely if _PyphenClass is guaranteed to be the class type or None:
    # def get_word_complexity_score(word: str, pyphen_dic: Optional[_PyphenClass] = None) -> int:
    # However, Pylance might still struggle if _PyphenClass definition is conditional. String forward reference is best.
    # Let's use a string forward reference for the Pyphen class type for robustness with linters
    # def get_word_complexity_score(word: str, pyphen_dic: "Optional[pyphen.Pyphen]" = None) -> int: 
    # Simpler for now if the above string forward ref is tricky due to pyphen not being defined globally
    # if PYPHEN_AVAILABLE is used to gate calling.
    # The most robust solution is to use the string literal for the type hint if the import is conditional.

    word_lower = word.lower()
    complexity = 0

    if len(word_lower) > 9: complexity += 3
    elif len(word_lower) > 6: complexity += 2
    elif len(word_lower) > 4: complexity += 1

    if PYPHEN_AVAILABLE and pyphen_dic: # Check both module availability and if dic was passed
        try:
            # pyphen_dic is an instance of the Pyphen class here
            syllables = len(pyphen_dic.inserted(word_lower).split('-'))
            if syllables > 3: complexity += 4
            elif syllables > 2: complexity += 2
            elif syllables > 1: complexity += 1
        except Exception as e:
            logger.debug(f"Pyphen syllable count error for '{word_lower}': {e}")
    
    very_common_words = {"the", "a", "is", "to", "and", "of", "it", "in", "on", "for", "be", "red", "blue", "dog", "cat"}
    if word_lower in very_common_words and len(word_lower) <= 4:
        complexity = max(0, complexity - 2) 

    return min(10, max(1, complexity))


def analyze_vocabulary(word_categories_data: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    logger.info("Starting vocabulary analysis...")
    enriched_vocabulary: Dict[str, Dict[str, Any]] = {}
    unique_words: Set[str] = set()

    for category, words in word_categories_data.items():
        for word in words:
            unique_words.add(word.lower())

    pyphen_dic_instance = None # Initialize to None
    if PYPHEN_AVAILABLE and _pyphen_module: # Check if the pyphen module was successfully imported
        try:
            # _PyphenClass here refers to pyphen.Pyphen if import was successful
            if _PyphenClass: # Check if the class itself was successfully obtained
                 pyphen_dic_instance = _PyphenClass(lang='en_US') 
                 logger.info("Pyphen dictionary initialized for syllable counting.")
            else:
                logger.warning("Pyphen class could not be referenced. Syllable counting skipped.")
        except Exception as e: 
            logger.warning(f"Could not initialize Pyphen object: {e}. Syllable counting will be skipped.")
    else:
        logger.warning("Pyphen library not available. Syllable counting will be skipped.")


    logger.info(f"Analyzing {len(unique_words)} unique words...")
    for word_lower in unique_words:
        original_category = "unknown"
        for category, words_in_cat in word_categories_data.items():
            if word_lower in [w.lower() for w in words_in_cat]:
                original_category = category
                break
        
        complexity_score = get_word_complexity_score(word_lower, pyphen_dic=pyphen_dic_instance)

        enriched_vocabulary[word_lower] = {
            "category": original_category,
            "complexity_score": complexity_score,
            "definition": "", 
            "example_sentence": "" 
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