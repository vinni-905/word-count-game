from fastapi import FastAPI, Request, HTTPException, Body, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field, validator, EmailStr
from datetime import timedelta # For token expiration

import logging
import os
import datetime as dt # Use dt to avoid conflict with datetime class from above
import time
import sqlite3
import json

# Import auth utilities from auth_utils.py (ensure this file exists)
from auth_utils import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token, # Used in get_current_user
    ACCESS_TOKEN_EXPIRE_MINUTES # Used in token creation
)

# Import functions from puzzle_logic module
try:
    from puzzle_logic import (
        generate_solvable_puzzle,
        check_puzzle_answer,
        get_puzzle_hint,
        get_or_generate_daily_challenge,
        active_puzzles,
        cleanup_old_puzzles
    )
    # Import from database module
    from database import (
        init_db,
        add_score_to_leaderboard,
        get_leaderboard_scores,
        add_user_submitted_puzzle,
        get_user_submitted_puzzles_by_status,
        update_user_submitted_puzzle_status,
        get_user_by_username, # NEW
        create_user_in_db,    # NEW
        get_user_by_id        # NEW (optional but good to have)
    )
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL IMPORT ERROR: {e}")
    raise SystemExit(f"Failed to import core logic or auth_utils: {e}")

# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="WordLinks Game")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

if not os.path.isdir(TEMPLATES_DIR):
    logger.error(f"Templates directory not found: {TEMPLATES_DIR}. HTML pages may not serve.")
if not os.path.isdir(STATIC_DIR):
    logger.warning(f"Static directory not found: {STATIC_DIR}. Creating it.")
    try:
        os.makedirs(STATIC_DIR, exist_ok=True)
        os.makedirs(os.path.join(STATIC_DIR, "sounds"), exist_ok=True)
        logger.info(f"Created static/sounds directory structure at {STATIC_DIR}")
    except OSError as e:
        logger.error(f"Could not create static directory: {e}")

templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Pydantic Models for Auth & Users ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserBase(BaseModel):
    username: Annotated[str, Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")] # Example pattern
    email: Optional[EmailStr] = None
    full_name: Optional[Annotated[str, Field(max_length=100)]] = None

class UserCreate(UserBase):
    password: Annotated[str, Field(min_length=8, max_length=100)]

class UserInDBBase(UserBase): # Base for DB representation
    id: int
    disabled: bool = False # In DB it's 0 or 1, Pydantic will convert

    class Config:
        from_attributes = True # Replaces orm_mode in Pydantic v2

class User(UserInDBBase): # For API responses to client
    pass

class UserInDB(UserInDBBase): # For internal use, includes hashed_password
    hashed_password: str


# --- Pydantic Models for Game Payloads ---
class UserPerformanceSummary(BaseModel):
    avg_hints: Optional[float] = None
    avg_mistakes: Optional[float] = None
    win_rate: Optional[float] = None
    avg_solve_time: Optional[float] = None
    plays: Optional[int] = None

class GenerateRequest(BaseModel):
    difficulty: str
    user_performance_summary: Optional[UserPerformanceSummary] = None

class HintRequest(BaseModel):
    puzzle_id: str
    solved_group_keys: Optional[List[str]] = []

class AnswerPayload(BaseModel):
    puzzle_id: str
    groups: Dict[str, List[str]]

class ScoreSubmitPayload(BaseModel):
    player_name: Annotated[str, Field(min_length=3, max_length=10)] # Keep for guest, override if logged in
    score: Annotated[int, Field(ge=0)]
    puzzle_difficulty: str
    time_taken: Optional[int] = None
    puzzle_id: Optional[str] = None
    game_mode: Optional[str] = "classic"

class UserPuzzleGroup(BaseModel):
    words: List[Annotated[str, Field(strip_whitespace=True, min_length=1, max_length=25)]]
    description: Annotated[str, Field(strip_whitespace=True, min_length=3, max_length=100)]

    @validator('words')
    def check_words_count_and_uniqueness(cls, v_words):
        if len(v_words) != 4:
            raise ValueError('Each group must contain exactly 4 words.')
        stripped_words = [word.strip() for word in v_words]
        if not all(stripped_words):
            raise ValueError('All words in a group must be non-empty.')
        if len(set(w.lower() for w in stripped_words)) != 4:
            raise ValueError('Words within a single group must be unique.')
        return stripped_words

class UserPuzzleSubmitPayload(BaseModel):
    submitter_name: Optional[Annotated[str, Field(strip_whitespace=True, max_length=50)]] = None
    group1: UserPuzzleGroup
    group2: UserPuzzleGroup
    group3: UserPuzzleGroup
    group4: UserPuzzleGroup

class PuzzleStatusUpdate(BaseModel):
    new_status: str


# --- Authentication Setup & Dependency ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

async def get_current_user_from_db(username: str) -> Optional[UserInDB]:
    user_dict = get_user_by_username(username)
    if user_dict:
        return UserInDB(**user_dict)
    return None

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    username = decode_access_token(token)
    if username is None:
        raise credentials_exception
    user = await get_current_user_from_db(username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: Annotated[UserInDB, Depends(get_current_user)]) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return User(**current_user.model_dump()) # Return User model for API responses

async def get_optional_current_user(token: Annotated[Optional[str], Depends(oauth2_scheme)]) -> Optional[UserInDB]:
    if not token:
        return None
    try:
        username = decode_access_token(token)
        if username is None: return None
        user = await get_current_user_from_db(username)
        if user is None or user.disabled: return None
        return user
    except Exception: # Includes JWTError or if user not found during validation by get_current_user
        return None


# --- FastAPI Event Handler ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing database...")
    init_db()
    time.sleep(0.5)
    logger.info("Cleaning up any old puzzles...")
    cleanup_old_puzzles()
    logger.info("Startup tasks complete.")


# --- HTML Serving Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request): # ... (same)
    logger.info("Serving home page (at /).")
    try:
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving /: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load home page.")

@app.get("/home.html", response_class=HTMLResponse)
async def read_home_explicit(request: Request): # ... (same)
    logger.info("Serving home page (at /home.html).")
    try:
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving /home.html: {e}", exc_info=True)
        raise HTTPException(status_code=500,detail="Could not load home page.")

@app.get("/game", response_class=HTMLResponse)
async def read_game_page(request: Request): # ... (same)
    logger.info("Serving game page (at /game).")
    try:
        return templates.TemplateResponse("game.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving /game: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load game page.")

@app.get("/submit-puzzle", response_class=HTMLResponse)
async def read_submit_puzzle_page(request: Request): # ... (same)
    logger.info("Serving user puzzle submission page.")
    try:
        return templates.TemplateResponse("submit_puzzle.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving /submit-puzzle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load puzzle submission page.")

@app.get("/admin/review-puzzles", response_class=HTMLResponse)
async def read_admin_review_page(request: Request): # ... (same)
    logger.info("Serving admin puzzle review page.")
    try:
        return templates.TemplateResponse("admin_puzzles.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving /admin/review-puzzles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load admin review page.")


# --- API Routes ---

# AUTHENTICATION ROUTES (NEW)
@app.post("/auth/token", response_model=Token)
async def login_for_access_token_api(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    logger.info(f"Login attempt for username: {form_data.username}")
    user = await get_current_user_from_db(form_data.username) # Fetches UserInDB
    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Login failed for user '{form_data.username}'. Incorrect username or password.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user.disabled:
        logger.warning(f"Login failed: User '{form_data.username}' is disabled.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    logger.info(f"User '{form_data.username}' logged in successfully.")
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user_api(user_in: UserCreate):
    logger.info(f"Registration attempt for username: {user_in.username}")
    db_user_check = await get_current_user_from_db(user_in.username)
    if db_user_check:
        logger.warning(f"Registration failed: Username '{user_in.username}' already registered.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    
    hashed_password = get_password_hash(user_in.password)
    user_id = create_user_in_db( # This function is synchronous
        username=user_in.username,
        hashed_password=hashed_password,
        email=user_in.email,
        full_name=user_in.full_name
    )
    if user_id is None:
        logger.error(f"Registration failed for '{user_in.username}' due to database error (likely username/email exists).")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username or email already exists, or database error.")
    
    created_user_dict = get_user_by_id(user_id) # Synchronous fetch
    if not created_user_dict:
        logger.error(f"Could not retrieve user '{user_in.username}' (ID: {user_id}) after creation.")
        raise HTTPException(status_code=500, detail="User created but could not be retrieved.")
        
    return User(**created_user_dict) # Return User model (without hashed_password)

@app.get("/users/me", response_model=User)
async def read_users_me_api(current_user: Annotated[User, Depends(get_current_active_user)]):
    return current_user

# --- Standard Game API Routes ---
@app.post("/api/generate_puzzle", response_class=JSONResponse)
async def api_generate_puzzle(request_data: GenerateRequest): # ... (same)
    difficulty = request_data.difficulty
    user_perf_summary_dict = request_data.user_performance_summary.model_dump() if request_data.user_performance_summary else None
    logger.info(f"API Req: Generate puzzle, Diff: {difficulty}, PerfSummary: {'Provided' if user_perf_summary_dict else 'None'}")
    if user_perf_summary_dict:
        logger.debug(f"Performance Summary Data: {user_perf_summary_dict}")
    try:
        if difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(status_code=400, detail="Invalid difficulty level selected.")
        puzzle_data = generate_solvable_puzzle(difficulty, user_performance_summary=user_perf_summary_dict)
        return JSONResponse(content=puzzle_data)
    except ValueError as e:
        logger.error(f"Gen ValErr: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Gen UnexpErr:")
        raise HTTPException(status_code=500, detail="Internal server error generating puzzle.")

@app.get("/api/daily_challenge", response_class=JSONResponse)
async def api_get_daily_challenge(): # ... (same)
    logger.info("API Req: Daily challenge.")
    try:
        daily_puzzle_data = get_or_generate_daily_challenge()
        return JSONResponse(content=daily_puzzle_data)
    except ValueError as e:
        logger.error(f"Daily ValErr: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Daily UnexpErr:")
        raise HTTPException(status_code=500, detail=f"Could not load daily challenge: {str(e)}")

@app.post("/api/check_answer", response_class=JSONResponse)
async def api_check_answer(payload: AnswerPayload): # ... (same)
    logger.info(f"API Req: Check answer, Puzzle: {payload.puzzle_id}")
    try:
        result = check_puzzle_answer(payload.puzzle_id, payload.groups)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"CheckAns Err {payload.puzzle_id}:")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/get_hint", response_class=JSONResponse)
async def api_get_hint(request_data: HintRequest): # ... (same)
    logger.info(f"API Req: Get hint, Puzzle: {request_data.puzzle_id}")
    try:
        result = get_puzzle_hint(request_data.puzzle_id, request_data.solved_group_keys)
        if result.get("hint") is None and "invalid" in result.get("message", "").lower():
             raise HTTPException(status_code=404, detail=result["message"])
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"Hint Err {request_data.puzzle_id}:")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/get_solution/{puzzle_id}", response_class=JSONResponse)
async def api_get_solution(puzzle_id: str): # ... (same)
    logger.info(f"API Req: Get solution, Puzzle: {puzzle_id}")
    if puzzle_id in active_puzzles:
        puzzle_data = active_puzzles[puzzle_id]
        solution_payload = { "groups": {} }
        descriptions = puzzle_data.get("descriptions", {})
        solution_map = puzzle_data.get("solution", {})
        parameters = puzzle_data.get("parameters", {})
        difficulty_index_map = parameters.get("difficulty_index_map", {})
        sorted_group_keys = sorted(solution_map.keys())
        for i, group_key in enumerate(sorted_group_keys):
            words = solution_map.get(group_key, [])
            difficulty_idx = difficulty_index_map.get(group_key, i)
            solution_payload["groups"][group_key] = {
                "description": descriptions.get(group_key, f"Group {i+1}"),
                "words": words,
                "difficulty_index": difficulty_idx
            }
        return JSONResponse(content=solution_payload)
    else:
        logger.warning(f"Solution for {puzzle_id} not found in active_puzzles.")
        raise HTTPException(status_code=404, detail="Puzzle solution not found or puzzle is no longer active.")

@app.get("/api/get_challenge_puzzle/{challenge_puzzle_id:str}", response_class=JSONResponse)
async def api_get_challenge_puzzle_data(challenge_puzzle_id: str): # ... (same)
    logger.info(f"API Req: Get challenge puzzle data for ID: {challenge_puzzle_id}")
    if challenge_puzzle_id in active_puzzles:
        puzzle_detail = active_puzzles[challenge_puzzle_id]
        logger.debug(f"Found challenge puzzle {challenge_puzzle_id} in active_puzzles.")
        return {
            "puzzle_id": puzzle_detail["puzzle_id"],
            "words": puzzle_detail.get("words_on_grid", []),
            "difficulty": puzzle_detail.get("difficulty", "Challenge")
        }
    else:
        logger.warning(f"Challenge puzzle {challenge_puzzle_id} not found in active_puzzles.")
        raise HTTPException(status_code=404, detail="Challenge puzzle not found or has expired.")

@app.post("/api/submit_user_puzzle", status_code=201)
async def api_submit_user_puzzle(
    payload: UserPuzzleSubmitPayload,
    current_user: Annotated[Optional[UserInDB], Depends(get_optional_current_user)] # MODIFIED
):
    user_id_to_store = current_user.id if current_user else None
    # Use provided submitter_name if available, otherwise username if logged in, else Anonymous
    submitter_display_name = payload.submitter_name.strip() if payload.submitter_name and payload.submitter_name.strip() \
                             else (current_user.username if current_user else "Anonymous")

    logger.info(f"API Req: User Puzzle Submission by {submitter_display_name} (User ID: {user_id_to_store})")
    all_submitted_words = []
    for i in range(1, 5):
        group_payload = getattr(payload, f"group{i}")
        all_submitted_words.extend([w.strip().lower() for w in group_payload.words])
    if len(set(all_submitted_words)) != 16:
        logger.warning(f"User puzzle submission failed: Not 16 unique words.")
        raise HTTPException(status_code=400, detail="All 16 words across all groups must be unique and non-empty.")
    try:
        submission_id = add_user_submitted_puzzle(
            submitter_name=submitter_display_name,
            user_id=user_id_to_store, # MODIFIED: Pass user_id
            group1_words=payload.group1.words, group1_description=payload.group1.description,
            group2_words=payload.group2.words, group2_description=payload.group2.description,
            group3_words=payload.group3.words, group3_description=payload.group3.description,
            group4_words=payload.group4.words, group4_description=payload.group4.description
        )
        if submission_id is None:
            raise HTTPException(status_code=500, detail="Could not save puzzle submission (database error or duplicate).")
        return {"message": "Puzzle submitted successfully! It will be reviewed.", "submission_id": submission_id}
    except ValueError as ve:
        logger.warning(f"User puzzle submission validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except sqlite3.Error as dbe:
        logger.error(f"Database error during user puzzle submission: {dbe}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error processing submission.")
    except Exception as e:
        logger.exception("Unexpected error during user puzzle submission:")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- Admin API Routes ---
@app.get("/api/admin/pending_puzzles", response_class=JSONResponse)
async def api_get_pending_puzzles_admin(): # ... (same)
    logger.info("Admin API Req: Get pending puzzles.")
    try:
        pending_puzzles = get_user_submitted_puzzles_by_status('pending')
        for puzzle in pending_puzzles:
            for i in range(1, 5):
                words_json_key = f'group{i}_words'
                if puzzle.get(words_json_key) and isinstance(puzzle[words_json_key], str):
                    try:
                        puzzle[words_json_key] = json.loads(puzzle[words_json_key])
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON for words in puzzle ID {puzzle.get('id')}, key {words_json_key}")
                        puzzle[words_json_key] = ["Error loading words"]
        return JSONResponse(content={"pending_puzzles": pending_puzzles})
    except Exception as e:
        logger.exception("Error fetching pending puzzles for admin:")
        raise HTTPException(status_code=500, detail="Could not fetch pending puzzles.")

@app.post("/api/admin/update_puzzle_status/{submission_id}", status_code=200)
async def api_update_puzzle_status_admin(submission_id: int, payload: PuzzleStatusUpdate): # ... (same)
    new_status = payload.new_status.lower()
    logger.info(f"Admin API Req: Update puzzle ID {submission_id} to status '{new_status}'.")
    if new_status not in ['approved', 'rejected']:
        raise HTTPException(status_code=400, detail="Invalid status. Must be 'approved' or 'rejected'.")
    success = update_user_submitted_puzzle_status(submission_id, new_status)
    if success:
        return {"message": f"Puzzle ID {submission_id} status updated to '{new_status}'."}
    else:
        raise HTTPException(status_code=404, detail=f"Could not update status for puzzle ID {submission_id}. It might not exist or a database error occurred.")

@app.post("/api/submit_score", status_code=201)
async def api_submit_score(
    payload: ScoreSubmitPayload,
    current_user: Annotated[Optional[UserInDB], Depends(get_optional_current_user)] # MODIFIED
):
    user_id_to_store = current_user.id if current_user else None
    # If user is logged in, use their username. Otherwise, use the name from payload.
    player_name_to_store = current_user.username if current_user else payload.player_name.strip()

    logger.info(f"API Req: Submit score for {player_name_to_store} (User ID: {user_id_to_store}), Score: {payload.score}, Diff: {payload.puzzle_difficulty}")
    try:
        add_score_to_leaderboard(
            player_name=player_name_to_store,
            score=payload.score,
            difficulty=payload.puzzle_difficulty,
            time_taken=payload.time_taken,
            puzzle_id=payload.puzzle_id,
            game_mode=payload.game_mode if payload.game_mode else "classic",
            user_id=user_id_to_store # MODIFIED: Pass user_id
        )
        return {"message": "Score submitted successfully!"}
    except Exception as e:
        logger.exception("Error submitting score:")
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while submitting the score: {str(e)}")
        raise

@app.get("/api/leaderboard", response_class=JSONResponse)
async def api_get_leaderboard( difficulty: Optional[str] = None, mode: Optional[str] = "classic", limit: Optional[int] = 10 ): # ... (same)
    logger.info(f"API Req: Get leaderboard. Diff: {difficulty}, Mode: {mode}, Limit: {limit}")
    try:
        safe_limit = max(1, min(limit if limit is not None else 10, 50))
        scores = get_leaderboard_scores(difficulty=difficulty, game_mode=mode if mode else "classic", limit=safe_limit)
        return {"leaderboard": scores}
    except Exception as e:
        logger.exception("Error fetching leaderboard:")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting WordLinks FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)