"""EDOPro bridge: lets WindBot play with the learning AI.

edopro_emu   what EDOPro's server sends each player (simulator side, for validation)
client_state the game state rebuilt from those messages (what the model sees in EDOPro)
session      decisions -> model -> responses, the same flow as sim.duel
server       the process WindBot starts: reads EDOPro messages, answers with responses
register     puts a model into WindBot/ and bots.json
"""
