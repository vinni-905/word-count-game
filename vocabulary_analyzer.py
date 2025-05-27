import json
import os
import logging
import time
from typing import Dict, List, Any, Set, Optional, Tuple, Type # Import Type for type hinting classes
from collections import defaultdict # For grouping words by category

# Attempt to import requests, pyphen. They are optional for basic functionality.
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

_PyphenClass: Optional[Type] = None # Store the Pyphen class type itself
PYPHEN_AVAILABLE = False
try:
    import pyphen as pyphen_module_imported
    if hasattr(pyphen_module_imported, 'Pyphen'):
        _PyphenClass = pyphen_module_imported.Pyphen # Assign the class, not an instance
        PYPHEN_AVAILABLE = True
except ImportError:
    pass

# --- Initialize Logger ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

if PYPHEN_AVAILABLE and _PyphenClass:
    logger.info("Pyphen module and Pyphen class are available.")
elif PYPHEN_AVAILABLE and not _PyphenClass:
    logger.warning("Pyphen module imported, but Pyphen class reference not found. Syllable counting may fail.")
else:
    logger.warning("Pyphen library not found or Pyphen class is not available. Syllable counting will be skipped.")

if REQUESTS_AVAILABLE:
    logger.info("Requests library available for API calls.")
else:
    logger.warning("Requests library not found. Definition fetching via API will be skipped.")


# --- Configuration ---
WORD_CATEGORIES_FOR_ANALYSIS: Dict[str, List[str]] = {
    'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'brown', 'gray', 'teal', 'maroon', 'indigo', 'violet', 'cyan', 'magenta', 'lime', 'olive', 'gold', 'silver', 'bronze', 'ivory', 'beige', 'scarlet', 'azure', 'amber', 'chartreuse', 'lavender', 'fuchsia', 'turquoise', 'khaki', 'ochre', 'sepia', 'crimson', 'viridian'],
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'snake', 'eagle', 'shark', 'whale', 'dolphin', 'penguin', 'owl', 'bat', 'bee', 'ant', 'aardvark', 'capybara', 'lemur', 'panda', 'koala', 'kangaroo', 'rhino', 'hippo', 'gorilla', 'ostrich', 'flamingo', 'peacock', 'swan', 'sloth', 'armadillo', 'meerkat', 'otter', 'badger', 'squirrel', 'chipmunk', 'beaver', 'hyena', 'jaguar', 'leopard', 'cheetah', 'cougar', 'lynx', 'coyote', 'bison', 'moose', 'elk', 'reindeer', 'camel', 'llama', 'alpaca', 'goat', 'sheep', 'pig', 'horse', 'donkey', 'mouse', 'rat', 'hamster', 'iguana', 'chameleon', 'gecko', 'turtle', 'tortoise', 'crocodile', 'alligator', 'frog', 'toad', 'salamander', 'newt', 'spider', 'scorpion', 'crab', 'lobster', 'shrimp', 'octopus', 'squid', 'jellyfish', 'starfish', 'seahorse', 'butterfly', 'moth', 'dragonfly', 'ladybug', 'grasshopper', 'cricket'],
    'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach', 'kiwi', 'watermelon', 'cherry', 'pear', 'blueberry', 'raspberry', 'lemon', 'plum', 'apricot', 'fig', 'date', 'guava', 'pomegranate', 'coconut', 'papaya', 'lychee', 'passionfruit', 'persimmon', 'cantaloupe', 'honeydew', 'nectarine', 'grapefruit', 'tangerine', 'clementine', 'boysenberry', 'cranberry', 'elderberry', 'gooseberry', 'mulberry', 'starfruit', 'dragonfruit'],
    'kitchen_items': ['knife', 'fork', 'spoon', 'plate', 'bowl', 'cup', 'pan', 'pot', 'oven', 'microwave', 'blender', 'toaster', 'kettle', 'grater', 'whisk', 'ladle', 'spatula', 'colander', 'strainer', 'peeler', 'corkscrew', 'can opener', 'measuring cup', 'measuring spoon', 'rolling pin', 'cutting board', 'tongs', 'masher', 'funnel', 'timer'],
    'vegetables': ['carrot', 'broccoli', 'spinach', 'potato', 'tomato', 'onion', 'garlic', 'lettuce', 'cucumber', 'pepper', 'celery', 'cabbage', 'cauliflower', 'eggplant', 'zucchini', 'asparagus', 'pea', 'bean', 'corn', 'mushroom', 'radish', 'beetroot', 'artichoke', 'pumpkin', 'squash', 'sweet potato', 'kale', 'leek', 'turnip', 'parsnip', 'okra'],
    'occupations': ['doctor', 'teacher', 'engineer', 'artist', 'chef', 'writer', 'musician', 'actor', 'firefighter', 'police', 'nurse', 'scientist', 'farmer', 'pilot', 'driver', 'builder', 'plumber', 'electrician', 'mechanic', 'dentist', 'lawyer', 'judge', 'accountant', 'architect', 'baker', 'barber', 'butcher', 'carpenter', 'cashier', 'cleaner', 'designer', 'director', 'editor', 'florist', 'gardener', 'hairdresser', 'journalist', 'librarian', 'manager', 'model', 'optician', 'painter', 'pharmacist', 'photographer', 'porter', 'postman', 'receptionist', 'sailor', 'salesperson', 'secretary', 'singer', 'soldier', 'surgeon', 'tailor', 'translator', 'veterinarian', 'waiter', 'welder'],
    'sports': ['soccer', 'basketball', 'tennis', 'baseball', 'golf', 'running', 'volleyball', 'swimming', 'boxing', 'skiing', 'cricket', 'rugby', 'cycling', 'karate', 'judo', 'archery', 'fencing', 'surfing', 'hockey', 'badminton', 'table tennis', 'wrestling', 'gymnastics', 'weightlifting', 'diving', 'rowing', 'sailing', 'snowboarding', 'skateboarding', 'climbing', 'fishing', 'bowling', 'darts', 'billiards', 'polo', 'sumo'],
    'musical_instruments': ['guitar', 'piano', 'violin', 'drums', 'trumpet', 'flute', 'saxophone', 'cello', 'clarinet', 'harp', 'ukulele', 'banjo', 'accordion', 'trombone', 'tuba', 'xylophone', 'harmonica', 'bagpipes', 'mandolin', 'bassoon', 'oboe', 'synthesizer', 'maracas', 'tambourine', 'triangle', 'cymbals'],
    'clothing': ['shirt', 'pants', 'dress', 'skirt', 'jacket', 'coat', 'sweater', 'hat', 'scarf', 'gloves', 'socks', 'shoes', 'boots', 'sandals', 'tie', 'belt', 'jeans', 'shorts', 'suit', 'blouse', 'hoodie', 'pajamas', 'robe', 'swimsuit', 'uniform', 'vest', 'tights', 'leggings', 'cap', 'beanie'],
    'emotions': ['happy', 'sad', 'angry', 'joy', 'fear', 'surprise', 'disgust', 'love', 'hate', 'hope', 'despair', 'anxiety', 'calm', 'excitement', 'boredom', 'envy', 'pride', 'shame', 'guilt', 'grief', 'relief', 'contentment', 'curiosity', 'trust', 'confusion', 'loneliness', 'jealousy'],
    'science_terms': ['atom', 'molecule', 'energy', 'gravity', 'photosynthesis', 'evolution', 'genetics', 'cell', 'dna', 'rna', 'protein', 'enzyme', 'ecosystem', 'habitat', 'fossil', 'mineral', 'velocity', 'acceleration', 'momentum', 'inertia', 'friction', 'magnetism', 'electricity', 'voltage', 'current', 'resistance', 'wavelength', 'frequency', 'amplitude', 'galaxy', 'planet', 'star', 'comet', 'asteroid', 'nebula', 'quasar', 'black hole', 'relativity', 'quantum'],
    'abstract_concepts': ['love', 'hate', 'justice', 'freedom', 'truth', 'beauty', 'knowledge', 'wisdom', 'power', 'chaos', 'order', 'peace', 'war', 'time', 'space', 'reality', 'dream', 'fantasy', 'logic', 'reason', 'faith', 'doubt', 'destiny', 'luck', 'fortune', 'fate', 'chance', 'hope', 'despair', 'courage', 'fear', 'happiness', 'sadness', 'success', 'failure', 'infinity', 'eternity', 'purpose', 'meaning', 'serendipity', 'ubiquitous'], # Added serendipity, ubiquitous
    'body_parts': ['head', 'hair', 'face', 'eye', 'ear', 'nose', 'mouth', 'tooth', 'tongue', 'neck', 'shoulder', 'arm', 'elbow', 'hand', 'finger', 'thumb', 'chest', 'stomach', 'back', 'waist', 'hip', 'leg', 'knee', 'foot', 'toe', 'ankle', 'heart', 'lung', 'liver', 'kidney', 'brain', 'bone', 'muscle', 'skin', 'vein', 'artery', 'nerve'],
    'transportation': ['car', 'bus', 'train', 'bicycle', 'motorcycle', 'airplane', 'boat', 'ship', 'truck', 'van', 'scooter', 'subway', 'tram', 'helicopter', 'rocket', 'taxi', 'ferry', 'yacht', 'canoe', 'kayak', 'skateboard', 'roller skates'],
    'weather': ['sun', 'rain', 'cloud', 'wind', 'snow', 'storm', 'fog', 'hail', 'mist', 'drizzle', 'thunder', 'lightning', 'hurricane', 'tornado', 'blizzard', 'monsoon', 'drought', 'flood', 'temperature', 'humidity', 'forecast', 'climate', 'season', 'rainbow', 'dew', 'frost']
}

MODEL_DIR = "model"
OUTPUT_FILENAME = os.path.join(MODEL_DIR, "enriched_word_data.json")

# --- API Function (Definition Fetching) ---
def get_definition_from_api(word: str) -> Tuple[Optional[str], Optional[str]]:
    if not REQUESTS_AVAILABLE or not requests:
        logger.debug(f"Requests library not available, skipping API call for '{word}'.")
        return None, None
    try:
        # Using a more specific user-agent can sometimes be helpful for public APIs
        headers = {'User-Agent': 'WordLinksGameAnalyzer/1.0 (your-email@example.com or contact info)'} # Optional
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", headers=headers, timeout=7) # Increased timeout
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            # Iterate through meanings to find one with definitions
            for entry in data:
                meanings = entry.get("meanings", [])
                if meanings and isinstance(meanings, list):
                    for meaning in meanings:
                        definitions = meaning.get("definitions", [])
                        if definitions and isinstance(definitions, list) and definitions[0]:
                            first_definition_obj = definitions[0]
                            definition = first_definition_obj.get("definition")
                            example = first_definition_obj.get("example") # Try to get example from the same definition
                            
                            # If no example there, try to find any example in this meaning
                            if not example:
                                for def_obj in definitions:
                                    if def_obj.get("example"):
                                        example = def_obj.get("example")
                                        break
                            
                            if definition: # Only return if a definition was found
                                if len(definition) > 250: definition = definition[:247] + "..." # Slightly longer truncation
                                if example and len(example) > 200: example = example[:197] + "..."
                                return definition, example
            logger.debug(f"No suitable definition structure found for '{word}' in API response.")
    except requests.exceptions.Timeout:
        logger.warning(f"API request timed out for word '{word}'.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.debug(f"Word '{word}' not found in API dictionary (404).")
        else:
            logger.warning(f"API HTTP error for word '{word}': {e}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"API request failed for word '{word}': {e}")
    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Error parsing API response for word '{word}': {e}")
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON response for word '{word}'. Error: {e}")
    return None, None

# --- Complexity Scoring Logic ---
def get_word_complexity_score(word: str, pyphen_dic: Optional[Any] = None) -> int:
    word_lower = word.lower()
    complexity = 0

    if not word_lower.isalpha(): # Penalize non-alpha words, or words with numbers/hyphens
        complexity += 2

    if len(word_lower) > 10: complexity += 4
    elif len(word_lower) > 7: complexity += 3
    elif len(word_lower) > 5: complexity += 2
    elif len(word_lower) > 3: complexity += 1


    if PYPHEN_AVAILABLE and pyphen_dic:
        try:
            syllables = len(pyphen_dic.inserted(word_lower).split('-'))
            if syllables > 4: complexity += 4
            elif syllables > 3: complexity += 3
            elif syllables > 2: complexity += 2
            elif syllables > 1: complexity += 1
        except Exception as e:
            logger.debug(f"Pyphen syllable count error for '{word_lower}': {e}")

    # Reduced list of very common words, more specific adjustment
    very_common_short_words = {"the", "a", "is", "to", "be", "in", "on", "it", "of", "and", "for"}
    common_short_words = {"cat", "dog", "sun", "run", "eat", "big", "red", "hot", "war", "car", "cup", "pan"}

    if word_lower in very_common_short_words:
        complexity = max(0, complexity - 3) # Stronger reduction
    elif word_lower in common_short_words:
        complexity = max(0, complexity - 1)

    # Bonus for rare letters (q, x, z, j) if not too short
    if len(word_lower) > 3:
        for char in ['q', 'x', 'z', 'j']:
            if char in word_lower:
                complexity += 1
                break # Add bonus only once

    return min(10, max(1, complexity + 1)) # Ensure score is at least 1, add 1 to base for non-zero start

# --- Main Analysis Function ---
def analyze_vocabulary(source_word_categories: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    logger.info("Starting vocabulary analysis...")
    
    # Load existing data if it exists, to potentially update or preserve manual edits
    # For now, we'll overwrite, but this is a place for future enhancement.
    # if os.path.exists(OUTPUT_FILENAME):
    #     try:
    #         with open(OUTPUT_FILENAME, 'r', encoding='utf-8') as f:
    #             enriched_vocabulary = json.load(f)
    #         logger.info(f"Loaded {len(enriched_vocabulary)} existing words from {OUTPUT_FILENAME}.")
    #     except Exception as e:
    #         logger.error(f"Could not load existing enriched data: {e}. Starting fresh.")
    #         enriched_vocabulary = {}
    # else:
    enriched_vocabulary: Dict[str, Dict[str, Any]] = {}

    unique_words: Set[str] = set()
    # Build initial word-to-category map and unique word set
    word_to_primary_category: Dict[str, str] = {}
    for category, words in source_word_categories.items():
        for word in words:
            lw = word.lower()
            unique_words.add(lw)
            if lw not in word_to_primary_category: # Assign first category encountered as primary
                word_to_primary_category[lw] = category


    pyphen_instance: Optional[Any] = None
    if PYPHEN_AVAILABLE and _PyphenClass:
        try:
            pyphen_instance = _PyphenClass(lang='en_US')
            logger.info("Pyphen dictionary object initialized for syllable counting.")
        except Exception as e:
            logger.warning(f"Could not initialize Pyphen object: {e}. Syllable counting will be skipped.")
    elif PYPHEN_AVAILABLE and not _PyphenClass:
         logger.warning("Pyphen module available, but Pyphen class could not be referenced. Syllable counting skipped.")

    logger.info(f"Analyzing {len(unique_words)} unique words...")
    processed_count = 0
    api_call_count = 0
    api_batch_size = 10 # How many API calls before a pause
    api_pause_duration = 0.5 # Seconds to pause

    for word_lower in sorted(list(unique_words)): # Process in alphabetical order for consistent output
        original_category = word_to_primary_category.get(word_lower, "unknown") # Get pre-assigned category

        # Check if word already exists and has definition (for re-runs, if you implement loading existing)
        # if word_lower in enriched_vocabulary and enriched_vocabulary[word_lower].get("definition"):
        #     complexity_score = get_word_complexity_score(word_lower, pyphen_dic=pyphen_instance) # Still update complexity
        #     enriched_vocabulary[word_lower]["complexity_score"] = complexity_score
        #     enriched_vocabulary[word_lower]["category"] = original_category # Update category in case it changed
        #     processed_count +=1
        #     continue # Skip API call if definition exists

        complexity_score = get_word_complexity_score(word_lower, pyphen_dic=pyphen_instance)

        definition, example = None, None
        if REQUESTS_AVAILABLE and requests:
            definition, example = get_definition_from_api(word_lower)
            api_call_count += 1
            if api_call_count % api_batch_size == 0:
                logger.info(f"API calls: Processed definitions for {api_call_count} words so far ({processed_count + 1} total words processed)...")
                time.sleep(api_pause_duration)
        
        enriched_vocabulary[word_lower] = {
            "category": original_category,
            "complexity_score": complexity_score,
            "definition": definition if definition else "",
            "example_sentence": example if example else ""
        }
        processed_count += 1
        if (processed_count % 200 == 0) or (processed_count == len(unique_words)):
            logger.info(f"Overall processing: {processed_count}/{len(unique_words)} words analyzed.")


    logger.info(f"Vocabulary analysis complete. Enriched {len(enriched_vocabulary)} words.")
    return enriched_vocabulary

if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logger.info(f"Created directory: {MODEL_DIR}")

    num_source_categories = len(WORD_CATEGORIES_FOR_ANALYSIS)
    num_source_words = sum(len(v) for v in WORD_CATEGORIES_FOR_ANALYSIS.values())

    if num_source_categories < 5 or num_source_words < 200 : # Increased threshold
        logger.warning(f"WORD_CATEGORIES_FOR_ANALYSIS seems small ({num_source_categories} categories, {num_source_words} words). Results may be limited. Please expand it for better analysis.")

    logger.info(f"Using WORD_CATEGORIES_FOR_ANALYSIS with {num_source_categories} categories and {num_source_words} words.")
    
    start_time = time.time()
    enriched_data = analyze_vocabulary(WORD_CATEGORIES_FOR_ANALYSIS)
    end_time = time.time()
    logger.info(f"Analysis took {end_time - start_time:.2f} seconds.")


    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=4, sort_keys=True)
        logger.info(f"Enriched vocabulary successfully saved to: {OUTPUT_FILENAME}")

        logger.info("\n--- Example Enriched Words (first 5 + specific checks) ---")
        display_count = 0
        specific_check_words = {"ubiquitous", "serendipity", "aardvark", "red", "apple", "zebra", "quantum"}
        
        # Print first N words
        for word_key in sorted(enriched_data.keys()): # Iterate in sorted order for consistency
            if display_count < 5:
                logger.info(f"'{word_key}': {enriched_data[word_key]}")
                display_count += 1
            else:
                break
        
        logger.info("--- Specific Word Checks ---")
        for s_word in specific_check_words:
            if s_word in enriched_data:
                logger.info(f"Specific check '{s_word}': {enriched_data[s_word]}")
            else:
                logger.info(f"Specific check '{s_word}': Not found in enriched_data (might not be in source categories).")

    except IOError as e:
        logger.error(f"Error writing enriched vocabulary to JSON file: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during saving or final logging: {e}", exc_info=True)