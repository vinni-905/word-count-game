from fastapi import FastAPI, Request, Form, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles # Not used if only serving templates
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
import os # Import os for path joining if needed, though Jinja2 handles it

# Import functions and state from puzzle_logic module
from puzzle_logic import (
    generate_solvable_puzzle,
    check_puzzle_answer,
    active_puzzles,
    get_puzzle_hint,
    cleanup_old_puzzles # Optional: If you want to add a cleanup endpoint/schedule
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="WordLinks Game")

# --- Request Models (Pydantic) ---
class HintRequest(BaseModel):
    puzzle_id: str
    solved_group_keys: Optional[List[str]] = []

class GenerateRequest(BaseModel):
    difficulty: str

class AnswerPayload(BaseModel):
    puzzle_id: str
    groups: Dict[str, List[str]] # Example: {"attempt_1": ["worda", "wordb"]}


# --- Templates ---
# Ensure your 'templates' directory is at the same level as main.py
# and contains home.html and game.html
# Project structure:
# your_project/
# ├── main.py
# ├── puzzle_logic.py
# └── templates/
#     ├── home.html
#     └── game.html

templates = Jinja2Templates(directory="templates")

# --- HTML Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the home page (difficulty selection) at the root URL."""
    logger.info("Serving home page (at /).")
    return templates.TemplateResponse("home.html", {"request": request})

# ****** ADD THIS ROUTE ******
@app.get("/home.html", response_class=HTMLResponse)
async def read_home_explicit(request: Request):
    """Serves the home page explicitly at /home.html."""
    logger.info("Serving home page (at /home.html).")
    return templates.TemplateResponse("home.html", {"request": request})
# ****************************

@app.get("/game", response_class=HTMLResponse) # This was correct
async def read_game(request: Request):
    """Serves the main game page."""
    logger.info("Serving game page.")
    return templates.TemplateResponse("game.html", {"request": request})

# --- API Routes ---
# (Your API routes remain the same - they look good)

@app.post("/api/generate_puzzle", response_class=JSONResponse)
async def api_generate_puzzle(request_data: GenerateRequest):
    """Generates a new puzzle based on selected difficulty."""
    difficulty = request_data.difficulty
    logger.info(f"Received request to generate puzzle with difficulty: {difficulty}")
    try:
        # Basic validation
        if difficulty not in ["easy", "medium", "hard"]:
             logger.warning(f"Invalid difficulty level received: {difficulty}")
             raise HTTPException(status_code=400, detail="Invalid difficulty level selected.")

        puzzle_data = generate_solvable_puzzle(difficulty)
        logger.info(f"Generated puzzle {puzzle_data.get('puzzle_id')} successfully.")
        return JSONResponse(content=puzzle_data)

    except ValueError as e:
         logger.error(f"ValueError during puzzle generation: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Error generating puzzle: {e}")
    except Exception as e:
        logger.exception("Unexpected error during puzzle generation:") # Logs traceback
        raise HTTPException(status_code=500, detail="Internal server error generating puzzle.")


@app.post("/api/check_answer", response_class=JSONResponse)
async def api_check_answer(payload: AnswerPayload):
    """Checks the user's submitted groups against the solution."""
    puzzle_id = payload.puzzle_id
    user_groups = payload.groups
    logger.info(f"Received answer check request for puzzle: {puzzle_id}")

    try:
        result = check_puzzle_answer(puzzle_id, user_groups)
        logger.info(f"Answer check result for {puzzle_id}: Correct={result.get('correct')}")
        return JSONResponse(content=result)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error during answer checking for puzzle {puzzle_id}:")
        raise HTTPException(status_code=500, detail="Internal server error checking answer.")


@app.post("/api/get_hint", response_class=JSONResponse)
async def api_get_hint(request_data: HintRequest):
    """Provides a hint for the given puzzle."""
    puzzle_id = request_data.puzzle_id
    logger.info(f"Received hint request for puzzle: {puzzle_id}")
    try:
        result = get_puzzle_hint(puzzle_id, request_data.solved_group_keys)
        if result.get("hint") is None and "invalid" in result.get("message", "").lower():
             logger.warning(f"Hint requested for invalid/expired puzzle ID: {puzzle_id}")
             raise HTTPException(status_code=404, detail=result["message"])
        elif result.get("hint") is None:
             logger.info(f"No hint provided for puzzle {puzzle_id}: {result['message']}")
             return JSONResponse(content={"hint": None, "message": result["message"]})
        else:
             logger.info(f"Hint provided successfully for puzzle {puzzle_id}")
             return JSONResponse(content=result)

    except HTTPException as e:
         raise e
    except Exception as e:
        logger.exception(f"Unexpected error getting hint for puzzle {puzzle_id}:")
        raise HTTPException(status_code=500, detail="Internal server error processing hint request.")

@app.get("/api/get_solution/{puzzle_id}", response_class=JSONResponse)
async def api_get_solution(puzzle_id: str):
    """Retrieves the full solution details for a given puzzle ID."""
    logger.info(f"Received request for solution for puzzle: {puzzle_id}")
    if puzzle_id in active_puzzles:
        puzzle_data = active_puzzles[puzzle_id]
        solution_payload = { "groups": {} }
        descriptions = puzzle_data.get("descriptions", {})
        solution_map = puzzle_data.get("solution", {})
        sorted_group_keys = sorted(solution_map.keys())

        for group_key in sorted_group_keys:
            words = solution_map[group_key]
            solution_payload["groups"][group_key] = {
                "description": descriptions.get(group_key, f"Group {group_key}"),
                "words": words,
            }
        logger.info(f"Solution retrieved successfully for puzzle: {puzzle_id}")
        return JSONResponse(content=solution_payload)
    else:
        logger.warning(f"Solution requested for non-active/invalid puzzle ID: {puzzle_id}")
        raise HTTPException(status_code=404, detail="Puzzle solution not found or puzzle is no longer active.")


# --- Server Runner ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting WordLinks server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) # Added reload=True for dev