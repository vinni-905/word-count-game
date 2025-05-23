import joblib
import os
from sklearn.pipeline import Pipeline # To check its type

MODEL_DIR = "model"
# Make sure this matches the name of your difficulty prediction model file
DIFFICULTY_MODEL_FILENAME = os.path.join(MODEL_DIR, "wordlinks_ridge_regression_model.pkl") 
PREPROCESSOR_FILENAME = os.path.join(MODEL_DIR, "feature_preprocessing.pkl")

print(f"--- Inspecting Difficulty Prediction Model ---")
if os.path.exists(DIFFICULTY_MODEL_FILENAME):
    try:
        loaded_object = joblib.load(DIFFICULTY_MODEL_FILENAME)
        print(f"Successfully loaded: {DIFFICULTY_MODEL_FILENAME}")
        print(f"Type of loaded object: {type(loaded_object)}")

        if isinstance(loaded_object, Pipeline):
            print("This IS a scikit-learn Pipeline object.")
            print("It likely contains preprocessing steps and the final estimator.")
            print("Pipeline steps:")
            for step_name, step_estimator in loaded_object.steps:
                print(f"  - Step Name: '{step_name}', Estimator: {type(step_estimator)}")
        else:
            print("This is NOT a scikit-learn Pipeline object.")
            print("It is likely just the regressor/classifier model itself.")
            print("You will probably need to load and apply 'feature_preprocessing.pkl' separately.")

    except Exception as e:
        print(f"Error loading or inspecting {DIFFICULTY_MODEL_FILENAME}: {e}")
else:
    print(f"ERROR: Difficulty model file not found at {DIFFICULTY_MODEL_FILENAME}")

print(f"\n--- Checking Preprocessor File ---")
if os.path.exists(PREPROCESSOR_FILENAME):
    try:
        preprocessor_object = joblib.load(PREPROCESSOR_FILENAME)
        print(f"Successfully loaded: {PREPROCESSOR_FILENAME}")
        print(f"Type of preprocessor object: {type(preprocessor_object)}")
        # You can add more checks here if you know what type it should be (e.g., ColumnTransformer)
        # from sklearn.compose import ColumnTransformer
        # if isinstance(preprocessor_object, ColumnTransformer):
        #     print("The preprocessor object IS a ColumnTransformer.")
    except Exception as e:
        print(f"Error loading or inspecting {PREPROCESSOR_FILENAME}: {e}")
else:
    print(f"NOTE: Preprocessor file not found at {PREPROCESSOR_FILENAME}. This is fine if your main model is a full pipeline.")