from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
import os
import datetime # For daily challenge

# Import functions from puzzle_logic module
# Ensure puzzle_logic.py is in the same directory or accessible
try:
    from puzzle_logic import (
        generate_solvable_puzzle,
        check_puzzle_answer,
        get_puzzle_hint,
        get_or_generate_daily_challenge, # Crucial import for daily challenge
        active_puzzles, # Used by get_solution endpoint
        cleanup_old_puzzles
    )
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL) # Basic config if logger not yet set
    logging.critical(f"CRITICAL ERROR: Failed to import 'puzzle_logic'. Ensure 'puzzle_logic.py' is correct and accessible. Error: {e}")
    # Exit if core logic is missing, as the app cannot function.
    raise SystemExit(f"Failed to import puzzle_logic: {e}")


# Configure logging (if not already configured by puzzle_logic, though it should be)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger for this module

app = FastAPI(title="WordLinks Game")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Check and create directories
if not os.path.isdir(TEMPLATES_DIR):
    logger.error(f"Templates directory not found: {TEMPLATES_DIR}. App may not serve HTML correctly.")
if not os.path.isdir(STATIC_DIR):
    logger.warning(f"Static directory not found: {STATIC_DIR}. Creating it.")
    try:
        os.makedirs(STATIC_DIR, exist_ok=True)
        os.makedirs(os.path.join(STATIC_DIR, "sounds"), exist_ok=True) # Ensure sounds subfolder
        logger.info(f"Created static/sounds directory structure at {STATIC_DIR}")
    except OSError as e:
        logger.error(f"Could not create static directory: {e}")


templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class HintRequest(BaseModel): puzzle_id: str; solved_group_keys: Optional[List[str]] = []
class GenerateRequest(BaseModel): difficulty: str
class AnswerPayload(BaseModel): puzzle_id: str; groups: Dict[str, List[str]]

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Cleaning up any old puzzles...")
    cleanup_old_puzzles() # Clean up puzzles on startup

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving home.html from /: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load home page.")

@app.get("/home.html", response_class=HTMLResponse)
async def read_home_explicit(request: Request):
    try:
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving home.html from /home.html: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load home page.")

@app.get("/game", response_class=HTMLResponse)
async def read_game(request: Request):
    try:
        return templates.TemplateResponse("game.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving game.html: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load game page.")


@app.post("/api/generate_puzzle", response_class=JSONResponse)
async def api_generate_puzzle(request_data: GenerateRequest):
    difficulty = request_data.difficulty
    logger.info(f"Received request to generate puzzle with difficulty: {difficulty}")
    try:
        if difficulty not in ["easy", "medium", "hard"]:
             raise HTTPException(status_code=400, detail="Invalid difficulty level selected.")
        puzzle_data = generate_solvable_puzzle(difficulty) # Calls your advanced logic
        return JSONResponse(content=puzzle_data)
    except ValueError as e: # Catch specific errors from puzzle_logic
        logger.error(f"ValueError during puzzle generation: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e)) # Send specific error to client
    except Exception as e:
        logger.exception("Unexpected error during puzzle generation:")
        raise HTTPException(status_code=500, detail="Internal server error generating puzzle.")

@app.get("/api/daily_challenge", response_class=JSONResponse)
async def api_get_daily_challenge():
    logger.info("Received request for daily challenge.")
    try:
        daily_puzzle_data = get_or_generate_daily_challenge() # Calls your advanced logic via daily wrapper
        return JSONResponse(content=daily_puzzle_data)
    except ValueError as e: # Catch specific errors from puzzle_logic
        logger.error(f"ValueError during daily challenge generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) # Send specific error to client
    except Exception as e:
        logger.exception("Error retrieving or generating daily challenge:")
        raise HTTPException(status_code=500, detail=f"Could not load daily challenge: {str(e)}")

@app.post("/api/check_answer", response_class=JSONResponse)
async def api_check_answer(payload: AnswerPayload):
    logger.info(f"Received answer check for puzzle: {payload.puzzle_id}")
    try:
        result = check_puzzle_answer(payload.puzzle_id, payload.groups)
        return JSONResponse(content=result)
    except Exception as e: # General catch-all
        logger.exception(f"Error checking answer for puzzle {payload.puzzle_id}:")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/get_hint", response_class=JSONResponse)
async def api_get_hint(request_data: HintRequest):
    logger.info(f"Received hint request for puzzle: {request_data.puzzle_id}")
    try:
        result = get_puzzle_hint(request_data.puzzle_id, request_data.solved_group_keys)
        if result.get("hint") is None and "invalid" in result.get("message", "").lower(): # Check if logic returned specific error
             raise HTTPException(status_code=404, detail=result["message"])
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"Error getting hint for puzzle {request_data.puzzle_id}:")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/get_solution/{puzzle_id}", response_class=JSONResponse)
async def api_get_solution(puzzle_id: str):
    logger.info(f"Received request for solution for puzzle: {puzzle_id}")
    if puzzle_id in active_puzzles:
        puzzle_data = active_puzzles[puzzle_id]
        solution_payload = { "groups": {} }
        descriptions = puzzle_data.get("descriptions", {})
        solution_map = puzzle_data.get("solution", {})
        difficulty_index_map = puzzle_data.get("parameters", {}).get("difficulty_index_map", {})

        sorted_group_keys = sorted(solution_map.keys())
        for i, group_key in enumerate(sorted_group_keys):
            words = solution_map[group_key]
            # Use the stored difficulty_index if available, otherwise fallback to enumeration order
            difficulty_idx = difficulty_index_map.get(group_key, i) # Default to loop index
            
            solution_payload["groups"][group_key] = {
                "description": descriptions.get(group_key, f"Group {i+1}"),
                "words": words, # These should be uppercase from puzzle_logic
                "difficulty_index": difficulty_idx
            }
        return JSONResponse(content=solution_payload)
    else:
        logger.warning(f"Solution requested for non-active/invalid puzzle ID: {puzzle_id}")
        raise HTTPException(status_code=404, detail="Puzzle solution not found or puzzle is no longer active.")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting WordLinks server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)