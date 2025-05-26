from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
import logging
import os
import datetime # Keep for other potential uses, though not directly for submission_date in add_score call
import time

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
    from database import init_db, add_score_to_leaderboard, get_leaderboard_scores
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL IMPORT ERROR: {e}")
    raise SystemExit(f"Failed to import core logic: {e}")

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

# --- Pydantic Models for Request Bodies ---
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
    player_name: Annotated[str, Field(min_length=3, max_length=10)]
    score: Annotated[int, Field(ge=0)]
    puzzle_difficulty: str
    time_taken: Optional[int] = None
    puzzle_id: Optional[str] = None
    game_mode: Optional[str] = "classic"

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
async def read_root(request: Request):
    logger.info("Serving home page (at /).")
    try:
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving /: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load home page.")

@app.get("/home.html", response_class=HTMLResponse)
async def read_home_explicit(request: Request):
    logger.info("Serving home page (at /home.html).")
    try:
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving /home.html: {e}", exc_info=True)
        raise HTTPException(status_code=500,detail="Could not load home page.")

@app.get("/game", response_class=HTMLResponse)
async def read_game_page(request: Request):
    logger.info("Serving game page (at /game).")
    try:
        return templates.TemplateResponse("game.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving /game: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load game page.")

# --- API Routes ---
@app.post("/api/generate_puzzle", response_class=JSONResponse)
async def api_generate_puzzle(request_data: GenerateRequest):
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
async def api_get_daily_challenge():
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
async def api_check_answer(payload: AnswerPayload):
    logger.info(f"API Req: Check answer, Puzzle: {payload.puzzle_id}")
    try:
        result = check_puzzle_answer(payload.puzzle_id, payload.groups)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"CheckAns Err {payload.puzzle_id}:")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/get_hint", response_class=JSONResponse)
async def api_get_hint(request_data: HintRequest):
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
async def api_get_solution(puzzle_id: str):
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

@app.post("/api/submit_score", status_code=201)
async def api_submit_score(payload: ScoreSubmitPayload):
    logger.info(f"API Req: Submit score for {payload.player_name}, Score: {payload.score}, Diff: {payload.puzzle_difficulty}")
    try:
        # submission_date is now handled by the database default
        add_score_to_leaderboard(
            player_name=payload.player_name.strip(),
            score=payload.score,
            difficulty=payload.puzzle_difficulty,
            time_taken=payload.time_taken,
            puzzle_id=payload.puzzle_id,
            game_mode=payload.game_mode if payload.game_mode else "classic"
            # No submission_date passed here
        )
        return {"message": "Score submitted successfully!"}
    except Exception as e:
        logger.exception("Error submitting score:")
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while submitting the score: {str(e)}")
        raise

@app.get("/api/leaderboard", response_class=JSONResponse)
async def api_get_leaderboard( difficulty: Optional[str] = None, mode: Optional[str] = "classic", limit: Optional[int] = 10 ):
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