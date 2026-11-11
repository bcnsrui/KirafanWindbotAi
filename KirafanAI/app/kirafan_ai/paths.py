import os

AI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the portable WindBot bridge lives inside WindBot/KirafanAI and is told where the game is
GAME_ROOT = os.environ.get("KIRAFAN_GAME_ROOT") or os.path.dirname(AI_ROOT)

OCGCORE_DLL = os.path.join(GAME_ROOT, "ocgcore.dll")
# installs that get the cards only from the EDOPro repository (config/user_configs.json) have no expansions copy
CARD_DB = next((p for p in (os.path.join(GAME_ROOT, "expansions", "kirafan.cdb"),
                            os.path.join(GAME_ROOT, "repositories", "KirafanTCG", "database", "kirafan.cdb"))
                if os.path.exists(p)), os.path.join(GAME_ROOT, "expansions", "kirafan.cdb"))
SCRIPT_DIRS = [
    os.path.join(GAME_ROOT, "expansions", "script"),
    os.path.join(GAME_ROOT, "script"),
    GAME_ROOT,
]
DECK_DIRS = [
    os.path.join(GAME_ROOT, "deck"),
    os.path.join(GAME_ROOT, "WindBot", "Decks"),
]
WINDBOT_DECK_DIR = os.path.join(GAME_ROOT, "WindBot", "Decks")
USER_DECK_DIR = os.path.join(GAME_ROOT, "deck")

PY32 = os.path.join(AI_ROOT, "py32", "python.exe")
RUNS_DIR = os.path.join(AI_ROOT, "runs")
