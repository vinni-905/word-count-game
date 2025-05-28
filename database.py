# database.py
import sqlite3
import logging
import os
import json
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

def init_db(force_recreate=False): # Added force_recreate flag
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if force_recreate:
            logger.warning(f"Forcing re-creation of tables in {DATABASE_NAME}. ALL EXISTING DATA IN THESE TABLES WILL BE LOST.")
            cursor.execute("DROP TABLE IF EXISTS scores")
            cursor.execute("DROP TABLE IF EXISTS user_submitted_puzzles")
            cursor.execute("DROP TABLE IF EXISTS users")
            logger.info("Existing tables (if any) dropped for re-creation.")

        # Users Table (Must be created before tables that reference it with FOREIGN KEY)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                email TEXT UNIQUE,
                full_name TEXT,
                disabled INTEGER DEFAULT 0,
                registration_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_username ON users (username)")

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
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_difficulty_score ON scores (difficulty, score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_mode_difficulty_score ON scores (game_mode, difficulty, score DESC)")
        # Only create this index if the users table and user_id column are confirmed to exist
        # This explicit check might not be necessary if tables are created in correct order
        # but added for robustness during debugging.
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id_scores ON scores (user_id)")
        except sqlite3.OperationalError as e:
            logger.error(f"Could not create index on scores(user_id), column might be missing or table schema issue: {e}")


        # User Submitted Puzzles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_submitted_puzzles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submitter_name TEXT,
                user_id INTEGER,
                group1_words TEXT NOT NULL,
                group1_description TEXT NOT NULL,
                group2_words TEXT NOT NULL,
                group2_description TEXT NOT NULL,
                group3_words TEXT NOT NULL,
                group3_description TEXT NOT NULL,
                group4_words TEXT NOT NULL,
                group4_description TEXT NOT NULL,
                submission_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_submitted_status ON user_submitted_puzzles (status)")
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_submitted_user_id ON user_submitted_puzzles (user_id)")
        except sqlite3.OperationalError as e:
            logger.error(f"Could not create index on user_submitted_puzzles(user_id), column might be missing: {e}")


        conn.commit()
        if force_recreate:
            logger.info(f"Database '{DATABASE_NAME}' tables RECREATED successfully at {DB_PATH}.")
        else:
            logger.info(f"Database '{DATABASE_NAME}' schema checked/initialized successfully at {DB_PATH}.")
    except sqlite3.Error as e:
        logger.error(f"Database error during initialization: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

# --- User related functions --- (Keep these as they are)
def get_user_by_username(username: str) -> Optional[Dict[str, Any]]: # ...
    conn = None
    user_data = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, hashed_password, email, full_name, disabled FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            user_data = dict(row)
    except sqlite3.Error as e:
        logger.error(f"Database error fetching user by username '{username}': {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
    return user_data

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]: # ...
    conn = None
    user_data = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, full_name, disabled FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            user_data = dict(row)
    except sqlite3.Error as e:
        logger.error(f"Database error fetching user by ID '{user_id}': {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
    return user_data

def create_user_in_db(username: str, hashed_password: str, email: Optional[str] = None, full_name: Optional[str] = None) -> Optional[int]: # ...
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (username, hashed_password, email, full_name)
            VALUES (?, ?, ?, ?)
        """, (username, hashed_password, email, full_name))
        conn.commit()
        user_id = cursor.lastrowid
        logger.info(f"User '{username}' created successfully with ID {user_id}.")
        return user_id
    except sqlite3.IntegrityError:
        logger.warning(f"Username '{username}' or email '{email}' already exists.")
        return None
    except sqlite3.Error as e:
        logger.error(f"Database error creating user '{username}': {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()

# --- Score related functions --- (Keep these as they are)
def add_score_to_leaderboard(player_name: str, score: int, difficulty: str,
                             time_taken: Optional[int] = None,
                             puzzle_id: Optional[str] = None,
                             game_mode: str = "classic",
                             user_id: Optional[int] = None): # ...
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scores (player_name, score, difficulty, time_taken_seconds, puzzle_id, game_mode, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (player_name, score, difficulty, time_taken, puzzle_id, game_mode, user_id))
        conn.commit()
        logger.info(f"Score added for {player_name} (User ID: {user_id or 'Guest'}): {score} (Diff: {difficulty}, Mode: {game_mode})")
    except sqlite3.Error as e:
        logger.error(f"Database error adding score for user {user_id or 'Guest'}: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def get_leaderboard_scores(difficulty: Optional[str] = None, game_mode: str = "classic", limit: int = 10) -> List[Dict[str, Any]]: # ...
    scores_list: List[Dict[str, Any]] = []
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT player_name, score, difficulty, time_taken_seconds, timestamp FROM scores WHERE game_mode = ?"
        params_list: List[Any] = [game_mode]
        if difficulty and difficulty in ["easy", "medium", "hard", "Daily Challenge", "challenge"]:
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

# --- User Submitted Puzzle functions --- (Keep these as they are)
def add_user_submitted_puzzle(
    group1_words: List[str], group1_description: str,
    group2_words: List[str], group2_description: str,
    group3_words: List[str], group3_description: str,
    group4_words: List[str], group4_description: str,
    submitter_name: Optional[str] = None,
    user_id: Optional[int] = None
) -> Optional[int]: # ...
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
                submitter_name, user_id,
                group1_words, group1_description,
                group2_words, group2_description,
                group3_words, group3_description,
                group4_words, group4_description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            submitter_name.strip() if submitter_name else None,
            user_id,
            g1w_json, group1_description.strip(),
            g2w_json, group2_description.strip(),
            g3w_json, group3_description.strip(),
            g4w_json, group4_description.strip()
        ))
        conn.commit()
        last_id = cursor.lastrowid
        logger.info(f"User puzzle (ID: {last_id}) submitted by '{submitter_name or 'Anonymous'}' (User ID: {user_id or 'Guest'}) added.")
        return last_id
    except sqlite3.Error as e:
        logger.error(f"Database error adding user puzzle for user {user_id or 'Guest'}: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()

def get_user_submitted_puzzles_by_status(status: str = 'pending') -> List[Dict[str, Any]]: # ...
    puzzles: List[Dict[str, Any]] = []
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, submitter_name, user_id,
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

def update_user_submitted_puzzle_status(submission_id: int, new_status: str) -> bool: # ...
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

# This function was missing from the puzzle_logic.py import error, so ensure it's here.
def get_random_approved_user_puzzle() -> Optional[Dict[str, Any]]:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, submitter_name, user_id,
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
                        puzzle_data[words_json_key] = json.loads(puzzle_data[words_json_key])
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON for words in user puzzle ID {puzzle_data.get('id')}, key {words_json_key}")
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


if __name__ == "__main__":
    print(f"Initializing/Updating database schema at: {DB_PATH} (FORCING RECREATION)")
    # Pass force_recreate=True if you want to guarantee tables are dropped and recreated
    # This is useful during development if schema changes frequently.
    # WARNING: THIS WILL DELETE ALL DATA IN THE TABLES.
    init_db(force_recreate=True)
    print("Database initialization/update (with potential forced recreation) complete.")