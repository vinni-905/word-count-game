# database.py
import sqlite3
import logging
import os
import json # For storing/loading list of words as JSON strings
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

DATABASE_NAME = "leaderboard.db"
DB_PATH = os.path.join(os.path.dirname(__file__), DATABASE_NAME)


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Scores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                score INTEGER NOT NULL,
                difficulty TEXT NOT NULL,
                time_taken_seconds INTEGER,
                puzzle_id TEXT,
                game_mode TEXT DEFAULT 'classic',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_difficulty_score ON scores (difficulty, score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_mode_difficulty_score ON scores (game_mode, difficulty, score DESC)")

        # User Submitted Puzzles Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_submitted_puzzles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submitter_name TEXT,
                group1_words TEXT NOT NULL,
                group1_description TEXT NOT NULL,
                group2_words TEXT NOT NULL,
                group2_description TEXT NOT NULL,
                group3_words TEXT NOT NULL,
                group3_description TEXT NOT NULL,
                group4_words TEXT NOT NULL,
                group4_description TEXT NOT NULL,
                submission_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending' -- e.g., pending, approved, rejected
            )
        """)
        conn.commit()
        logger.info(f"Database '{DATABASE_NAME}' initialized/updated with tables successfully at {DB_PATH}.")
    except sqlite3.Error as e:
        logger.error(f"Database error during initialization: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def add_score_to_leaderboard(player_name: str, score: int, difficulty: str,
                             time_taken: Optional[int] = None,
                             puzzle_id: Optional[str] = None,
                             game_mode: str = "classic"):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scores (player_name, score, difficulty, time_taken_seconds, puzzle_id, game_mode)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (player_name, score, difficulty, time_taken, puzzle_id, game_mode))
        conn.commit()
        logger.info(f"Score added for {player_name}: {score} (Diff: {difficulty}, Mode: {game_mode})")
    except sqlite3.Error as e:
        logger.error(f"Database error adding score: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def add_user_submitted_puzzle(
    group1_words: List[str], group1_description: str,
    group2_words: List[str], group2_description: str,
    group3_words: List[str], group3_description: str,
    group4_words: List[str], group4_description: str,
    submitter_name: Optional[str] = None
) -> Optional[int]:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        g1w_json = json.dumps(sorted([w.strip().upper() for w in group1_words]))
        g2w_json = json.dumps(sorted([w.strip().upper() for w in group2_words]))
        g3w_json = json.dumps(sorted([w.strip().upper() for w in group3_words]))
        g4w_json = json.dumps(sorted([w.strip().upper() for w in group4_words]))

        cursor.execute("""
            INSERT INTO user_submitted_puzzles (
                submitter_name,
                group1_words, group1_description,
                group2_words, group2_description,
                group3_words, group3_description,
                group4_words, group4_description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            submitter_name.strip() if submitter_name else None,
            g1w_json, group1_description.strip(),
            g2w_json, group2_description.strip(),
            g3w_json, group3_description.strip(),
            g4w_json, group4_description.strip()
        ))
        conn.commit()
        last_id = cursor.lastrowid
        logger.info(f"User puzzle (ID: {last_id}) submitted by '{submitter_name or 'Anonymous'}' added to database with status 'pending'.")
        return last_id
    except sqlite3.Error as e:
        logger.error(f"Database error adding user puzzle: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()

def get_user_submitted_puzzles_by_status(status: str = 'pending') -> List[Dict[str, Any]]:
    puzzles: List[Dict[str, Any]] = []
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, submitter_name, 
                   group1_words, group1_description,
                   group2_words, group2_description,
                   group3_words, group3_description,
                   group4_words, group4_description,
                   submission_date, status
            FROM user_submitted_puzzles
            WHERE status = ?
            ORDER BY submission_date DESC
        """, (status,))
        rows = cursor.fetchall()
        for row in rows:
            puzzle_data = dict(row)
            puzzles.append(puzzle_data)
        logger.info(f"Fetched {len(puzzles)} user-submitted puzzles with status '{status}'.")
    except sqlite3.Error as e:
        logger.error(f"Database error fetching user-submitted puzzles: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
    return puzzles

def update_user_submitted_puzzle_status(submission_id: int, new_status: str) -> bool:
    if new_status not in ['pending', 'approved', 'rejected']:
        logger.warning(f"Invalid status update: '{new_status}' for submission ID {submission_id}.")
        return False
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE user_submitted_puzzles
            SET status = ?
            WHERE id = ?
        """, (new_status, submission_id))
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Status of user puzzle ID {submission_id} updated to '{new_status}'.")
            return True
        else:
            logger.warning(f"No puzzle found with ID {submission_id} to update status.")
            return False
    except sqlite3.Error as e:
        logger.error(f"Database error updating status for puzzle ID {submission_id}: {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()

def get_random_approved_user_puzzle() -> Optional[Dict[str, Any]]:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, submitter_name, 
                   group1_words, group1_description,
                   group2_words, group2_description,
                   group3_words, group3_description,
                   group4_words, group4_description
            FROM user_submitted_puzzles
            WHERE status = 'approved'
            ORDER BY RANDOM() 
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            puzzle_data = dict(row)
            for i in range(1, 5):
                words_json_key = f'group{i}_words'
                if puzzle_data.get(words_json_key) and isinstance(puzzle_data[words_json_key], str):
                    try:
                        loaded_words = json.loads(puzzle_data[words_json_key])
                        if isinstance(loaded_words, list) and all(isinstance(w, str) for w in loaded_words):
                            puzzle_data[words_json_key] = loaded_words
                        else:
                            logger.error(f"Decoded words for {words_json_key} in user puzzle ID {puzzle_data.get('id')} is not a list of strings. Content: {loaded_words}")
                            return None
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON for words in user puzzle ID {puzzle_data.get('id')}, key {words_json_key}: {e}")
                        return None
                elif not isinstance(puzzle_data.get(words_json_key), list):
                    logger.error(f"Words for {words_json_key} in user puzzle ID {puzzle_data.get('id')} is not stored as a valid JSON string or is malformed.")
                    return None
                elif puzzle_data.get(words_json_key) is None:
                     logger.error(f"Words for {words_json_key} in user puzzle ID {puzzle_data.get('id')} is NULL in database.")
                     return None
            logger.info(f"Fetched approved user puzzle ID: {puzzle_data.get('id')}")
            return puzzle_data
        else:
            logger.info("No approved user puzzles found in the database.")
            return None
    except sqlite3.Error as e:
        logger.error(f"Database error fetching random approved user puzzle: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()

def get_leaderboard_scores(difficulty: Optional[str] = None, game_mode: str = "classic", limit: int = 10) -> List[Dict[str, Any]]:
    scores_list: List[Dict[str, Any]] = []
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT player_name, score, difficulty, time_taken_seconds, timestamp FROM scores WHERE game_mode = ?"
        params_list: List[Any] = [game_mode]
        if difficulty and difficulty in ["easy", "medium", "hard", "Daily Challenge"]:
            query += " AND difficulty = ?"
            params_list.append(difficulty)
        query += " ORDER BY score DESC, time_taken_seconds ASC, timestamp ASC LIMIT ?"
        params_list.append(limit)
        cursor.execute(query, tuple(params_list))
        rows = cursor.fetchall()
        for row in rows:
            scores_list.append(dict(row))
        logger.info(f"Fetched {len(scores_list)} scores for mode '{game_mode}' diff '{difficulty}' limit {limit}")
    except sqlite3.Error as e:
        logger.error(f"Database error fetching leaderboard: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
    return scores_list

if __name__ == "__main__":
    print(f"Initializing/Updating database schema at: {DB_PATH}")
    init_db()
    print("Database initialization/update complete.")

    # --- TEST CODE FOR get_random_approved_user_puzzle (uncomment to test) ---
    # print("\nAttempting to add a test user puzzle for integration testing...")
    # try:
    #     # This part is only to ensure there's an approved puzzle for testing.
    #     # In a real scenario, approval happens via the admin interface.
    #     conn = get_db_connection()
    #     cursor = conn.cursor()
    #     cursor.execute("SELECT id FROM user_submitted_puzzles WHERE submitter_name = 'DevTestUser' AND group1_description = 'Test Fruits'")
    #     existing = cursor.fetchone()
    #     if not existing:
    #         test_puzzle_id_temp = add_user_submitted_puzzle(
    #             submitter_name="DevTestUser",
    #             group1_words=["APPLE", "PEAR", "PLUM", "FIG"], group1_description="Test Fruits",
    #             group2_words=["CAR", "BUS", "TRAIN", "BIKE"], group2_description="Test Vehicles",
    #             group3_words=["RED", "BLUE", "GREEN", "BLACK"], group3_description="Test Colors",
    #             group4_words=["ONE", "TWO", "THREE", "FOUR"], group4_description="Test Numbers"
    #         )
    #         if test_puzzle_id_temp:
    #             update_user_submitted_puzzle_status(test_puzzle_id_temp, "approved")
    #             print(f"Added and approved a test user puzzle with ID: {test_puzzle_id_temp}")
    #     else:
    #         # Ensure it's approved if it exists
    #         update_user_submitted_puzzle_status(existing['id'], "approved")
    #         print(f"Test user puzzle ID {existing['id']} already exists and ensured it is approved.")
    # except sqlite3.Error as e:
    #     print(f"Error adding/approving test puzzle: {e}")
    # finally:
    #     if conn: # conn might not be defined if get_db_connection failed earlier
    #         conn.close()

    # print("\nFetching a random approved user puzzle:")
    # approved_puzzle = get_random_approved_user_puzzle()
    # if approved_puzzle:
    #     print(f"  Successfully fetched approved puzzle ID: {approved_puzzle.get('id')}")
    #     print(f"  Submitter: {approved_puzzle.get('submitter_name', 'N/A')}")
    #     for i in range(1, 5):
    #         words = approved_puzzle.get(f'group{i}_words', [])
    #         desc = approved_puzzle.get(f'group{i}_description', 'N/A')
    #         print(f"  Group {i}: {words} - '{desc}'")
    # else:
    #     print("  No approved user puzzle found, or an error occurred during fetch.")
    # --- END TEST CODE ---