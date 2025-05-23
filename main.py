from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
import os
import datetime

try:
    from puzzle_logic import (
        generate_solvable_puzzle,
        check_puzzle_answer,
        get_puzzle_hint,
        get_or_generate_daily_challenge,
        active_puzzles,
        cleanup_old_puzzles
    )
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: Failed to import 'puzzle_logic'. Error: {e}")
    raise SystemExit(f"Failed to import puzzle_logic: {e}")

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="WordLinks Game")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.isdir(TEMPLATES_DIR): logger.error(f"Templates dir not found: {TEMPLATES_DIR}")
if not os.path.isdir(STATIC_DIR):
    logger.warning(f"Static dir not found: {STATIC_DIR}, creating...")
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(os.path.join(STATIC_DIR, "sounds"), exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class UserPerformanceSummary(BaseModel):
    avg_hints: Optional[float] = None
    avg_mistakes: Optional[float] = None
    win_rate: Optional[float] = None
    avg_solve_time: Optional[float] = None
    plays: Optional[int] = None

class GenerateRequest(BaseModel):
    difficulty: str
    user_performance_summary: Optional[UserPerformanceSummary] = None

class HintRequest(BaseModel): puzzle_id: str; solved_group_keys: Optional[List[str]] = []
class AnswerPayload(BaseModel): puzzle_id: str; groups: Dict[str, List[str]]

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Cleaning up any old puzzles...")
    cleanup_old_puzzles()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try: return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e: logger.error(f"Error serving /: {e}", exc_info=True); raise HTTPException(500, "Could not load home page.")
@app.get("/home.html", response_class=HTMLResponse)
async def read_home_explicit(request: Request):
    try: return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e: logger.error(f"Error serving /home.html: {e}", exc_info=True); raise HTTPException(500,"Could not load home page.")
@app.get("/game", response_class=HTMLResponse)
async def read_game_page(request: Request):
    try: return templates.TemplateResponse("game.html", {"request": request})
    except Exception as e: logger.error(f"Error serving /game: {e}", exc_info=True); raise HTTPException(500, "Could not load game page.")

@app.post("/api/generate_puzzle", response_class=JSONResponse)
async def api_generate_puzzle(request_data: GenerateRequest): # Updated to use GenerateRequest
    difficulty = request_data.difficulty
    user_perf_summary_dict = request_data.user_performance_summary.model_dump() if request_data.user_performance_summary else None
    
    logger.info(f"API Req: Generate puzzle, Diff: {difficulty}, Perf: {user_perf_summary_dict}")
    
    try:
        if difficulty not in ["easy", "medium", "hard"]: 
            raise HTTPException(status_code=400, detail="Invalid difficulty level selected.")
        
        # Pass the performance summary dictionary to puzzle_logic
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
    except ValueError as e: logger.error(f"Daily ValErr: {e}"); raise HTTPException(500, str(e))
    except Exception as e: logger.exception("Daily UnexpErr:"); raise HTTPException(500, "Could not load daily.")

@app.post("/api/check_answer", response_class=JSONResponse)
async def api_check_answer(payload: AnswerPayload):
    logger.info(f"API Req: Check answer, Puzzle: {payload.puzzle_id}")
    try:
        result = check_puzzle_answer(payload.puzzle_id, payload.groups)
        return JSONResponse(content=result)
    except Exception as e: logger.exception(f"CheckAns Err {payload.puzzle_id}:"); raise HTTPException(500, str(e))

@app.post("/api/get_hint", response_class=JSONResponse)
async def api_get_hint(request_data: HintRequest):
    logger.info(f"API Req: Get hint, Puzzle: {request_data.puzzle_id}")
    try:
        result = get_puzzle_hint(request_data.puzzle_id, request_data.solved_group_keys)
        if result.get("hint") is None and "invalid" in result.get("message", "").lower():
             raise HTTPException(status_code=404, detail=result["message"])
        return JSONResponse(content=result)
    except Exception as e: logger.exception(f"Hint Err {request_data.puzzle_id}:"); raise HTTPException(500, str(e))

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
                "words": words, "difficulty_index": difficulty_idx
            }
        return JSONResponse(content=solution_payload)
    else:
        logger.warning(f"Solution for {puzzle_id} not found.")
        raise HTTPException(status_code=404, detail="Puzzle solution not found.")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting WordLinks FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)