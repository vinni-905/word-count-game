# database.py
import sqlite3
import logging
import os
# from datetime import datetime # Not strictly needed here anymore if DB handles timestamp
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
    conn = None # Initialize conn to None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Table name is 'scores'
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
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_difficulty_score ON scores (difficulty, score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_mode_difficulty_score ON scores (game_mode, difficulty, score DESC)")
        conn.commit()
        logger.info(f"Database '{DATABASE_NAME}' initialized successfully at {DB_PATH}.")
    except sqlite3.Error as e:
        logger.error(f"Database error during initialization: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

# MODIFIED: Removed submission_date from arguments as DB will handle it with DEFAULT CURRENT_TIMESTAMP
def add_score_to_leaderboard(player_name: str, score: int, difficulty: str,
                             time_taken: Optional[int] = None,
                             puzzle_id: Optional[str] = None,
                             game_mode: str = "classic"):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # MODIFIED: Removed timestamp from INSERT statement; DB default will be used.
        # The column name in the table is `timestamp`.
        # The argument name in the Pydantic model was `time_taken_seconds`,
        # but in the database it is `time_taken_seconds`.
        # Let's ensure we are inserting into the correct column name `time_taken_seconds`.
        # The `puzzle_id` and `game_mode` were added for completeness if your schema needs them.
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

def get_leaderboard_scores(difficulty: Optional[str] = None, game_mode: str = "classic", limit: int = 10) -> List[Dict[str, Any]]:
    scores_list: List[Dict[str, Any]] = [] # Renamed for clarity from 'scores'
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Column name in DB is `time_taken_seconds` and `timestamp`
        query = "SELECT player_name, score, difficulty, time_taken_seconds, timestamp FROM scores WHERE game_mode = ?"
        params_list: List[Any] = [game_mode]

        if difficulty and difficulty in ["easy", "medium", "hard", "Daily Challenge"]:
            query += " AND difficulty = ?"
            params_list.append(difficulty)

        query += " ORDER BY score DESC, time_taken_seconds ASC, timestamp ASC LIMIT ?" # Added timestamp for tie-breaking
        params_list.append(limit)

        cursor.execute(query, tuple(params_list))
        rows = cursor.fetchall()
        for row in rows:
            scores_list.append(dict(row)) # Use the renamed list

        logger.info(f"Fetched {len(scores_list)} scores for mode '{game_mode}' diff '{difficulty}' limit {limit}")
    except sqlite3.Error as e:
        logger.error(f"Database error fetching leaderboard: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
    return scores_list # Return the renamed list

if __name__ == "__main__":
    print(f"Initializing database at: {DB_PATH}")
    init_db()
    print("Database initialization complete (if not already done).")

    # Example usage (optional - for testing this script directly)
    # print("\nAdding some test scores...")
    # add_score_to_leaderboard(player_name="Alice", score=1500, difficulty="medium", time_taken=120, game_mode="classic")
    # add_score_to_leaderboard(player_name="Bob", score=1200, difficulty="medium", time_taken=150, game_mode="classic")
    # add_score_to_leaderboard(player_name="Charlie", score=1800, difficulty="hard", time_taken=100, game_mode="classic")
    # add_score_to_leaderboard(player_name="Diana", score=1500, difficulty="medium", time_taken=110, game_mode="classic") # Better time than Alice

    # print("\nFetching medium scores:")
    # medium_scores = get_leaderboard_scores(difficulty="medium", game_mode="classic", limit=5)
    # for s in medium_scores:
    #     print(f"  {s['player_name']}: {s['score']} (Time: {s['time_taken_seconds']}s, Submitted: {s['timestamp']})")

    # print("\nFetching all classic scores:")
    # all_classic_scores = get_leaderboard_scores(game_mode="classic", limit=5)
    # for s in all_classic_scores:
    #     print(f"  {s['player_name']}: {s['score']} (Diff: {s['difficulty']}, Time: {s['time_taken_seconds']}s, Submitted: {s['timestamp']})")