# convert_glove_to_w2v.py
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- Configuration ---
# Path to the downloaded GloVe .txt file (AFTER UNZIPPING)
# This path is set based on your input "C:\Users\own\Downloads\glove.6B"
# and assumes glove.6B.100d.txt is inside that folder.
GLOVE_INPUT_FILE = "C:/Users/own/Downloads/glove.6B/glove.6B.100d.txt"
# Note: Using forward slashes for better cross-platform compatibility in Python strings,
# but r"C:\Users\own\Downloads\glove.6B\glove.6B.100d.txt" would also work on Windows.

# This assumes your 'model' folder is in the same directory as this script
# (i.e., your project root: C:\Users\own\Downloads\word_links 3-2 project\model\)
MODEL_OUTPUT_DIR = "model" 
WORD2VEC_TEXT_OUTPUT_FILENAME = "glove.6B.100d.word2vec.txt" # Intermediate file, will be deleted
KEYED_VECTORS_BINARY_OUTPUT_FILENAME = "glove.6B.100d.kv" # Final, efficient format

def convert():
    # Check if the GLOVE_INPUT_FILE exists
    if not os.path.exists(GLOVE_INPUT_FILE):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: GloVe input file NOT FOUND at the specified path: '{GLOVE_INPUT_FILE}'")
        print(f"Please ensure the file 'glove.6B.100d.txt' exists at that exact location.")
        print(f"If you unzipped 'glove.6B.zip' into 'C:\\Users\\own\\Downloads\\', make sure")
        print(f"the folder structure is 'C:\\Users\\own\\Downloads\\glove.6B\\glove.6B.100d.txt'.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        print(f"Created output directory: {MODEL_OUTPUT_DIR}")

    word2vec_text_output_path = os.path.join(MODEL_OUTPUT_DIR, WORD2VEC_TEXT_OUTPUT_FILENAME)
    keyed_vectors_binary_output_path = os.path.join(MODEL_OUTPUT_DIR, KEYED_VECTORS_BINARY_OUTPUT_FILENAME)

    # Check if the final .kv file already exists to avoid re-processing
    if os.path.exists(keyed_vectors_binary_output_path):
        print(f"Output file '{keyed_vectors_binary_output_path}' already exists. Skipping conversion.")
        print(f"If you want to re-convert, please delete '{keyed_vectors_binary_output_path}' first.")
        return

    # 1. Convert GloVe format to Word2Vec text format
    print(f"\nConverting '{GLOVE_INPUT_FILE}'")
    print(f"To intermediate Word2Vec text format: '{word2vec_text_output_path}'...")
    try:
        glove2word2vec(GLOVE_INPUT_FILE, word2vec_text_output_path)
        print("Conversion to Word2Vec text format successful.")
    except Exception as e:
        print(f"Error during glove2word2vec conversion: {e}")
        print(f"Please check the path to your GloVe file and ensure it is a valid GloVe .txt file.")
        return

    # 2. Load the Word2Vec text format and save as efficient KeyedVectors binary format
    print(f"\nLoading '{word2vec_text_output_path}'")
    print(f"And saving as efficient KeyedVectors binary format: '{keyed_vectors_binary_output_path}'...")
    try:
        model = KeyedVectors.load_word2vec_format(word2vec_text_output_path, binary=False)
        model.save(keyed_vectors_binary_output_path) 
        print(f"Successfully saved KeyedVectors to: {keyed_vectors_binary_output_path}")
        print(f"\nIMPORTANT: You should now update 'puzzle_logic.py' to use the filename:")
        print(f"   WORD2VEC_ACTUAL_FILENAME = \"{KEYED_VECTORS_BINARY_OUTPUT_FILENAME}\"")
        print(f"   And ensure KeyedVectors.load() is used for it.")

    except Exception as e:
        print(f"Error loading Word2Vec text format or saving KeyedVectors: {e}")
    finally:
        # Clean up the intermediate .txt file
        if os.path.exists(word2vec_text_output_path):
            try:
                os.remove(word2vec_text_output_path)
                print(f"Cleaned up intermediate file: {word2vec_text_output_path}")
            except OSError as e:
                print(f"Error deleting intermediate file {word2vec_text_output_path}: {e}")
    
if __name__ == "__main__":
    print("--- GloVe to Gensim KeyedVectors (.kv) Conversion Script ---")
    print(f"This script will attempt to convert the GloVe file specified by GLOVE_INPUT_FILE.")
    print(f"Current GLOVE_INPUT_FILE path: '{GLOVE_INPUT_FILE}'")
    print(f"The output (.kv file) will be saved in your project's '{MODEL_OUTPUT_DIR}' folder.")
    print("Ensure 'gensim' is installed (`pip install gensim`).")
    print("This process might take a few minutes depending on the file size and your system.\n")
    
    confirmation = input("Proceed with conversion using the path above? (yes/no): ").strip().lower()
    if confirmation == 'yes':
        convert()
    else:
        print("Conversion aborted by user. Please update GLOVE_INPUT_FILE in the script if needed.")