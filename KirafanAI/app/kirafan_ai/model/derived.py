"""Game-knowledge features computed from a view (docs/heuristic_guide.md).

Shared by the network encoder (input features), the reward shaping and the rule-based AI,
so the three always agree on what "advantage" and "about to lose" mean.

Sanctuary level (성전 레벨) is the sum of levels of a player's own fallen creamates
(face-up cards in their extra deck). During the realist time the turn player loses at their
end phase if their own sanctuary level >= the opponent's (Kirafan.TurnPositionop), so a LOWER
own sanctuary level is better:  advantage = opp_sanctuary - my_sanctuary.
"""
from ..core.constants import (LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE,
                              LOCATION_EXTRA, POS_FACEUP, POS_DEFENSE, PHASE_MAIN1, PHASE_BATTLE_START,
                              PHASE_BATTLE_STEP, PHASE_DAMAGE, PHASE_DAMAGE_CAL, PHASE_BATTLE,
                              COUNTER_ATTACKS, COUNTER_WEAK, COUNTER_BIND)

REALIST_LEVEL = 10      # Kirafan.RealistTime: opponent sanctuary level >= 10
MID_GAME_STARLIGHT = 8  # guide 4-3: starlight (성채석) >= 8 is mid game
HAND_CAP = 5            # non-turn player discards down to 5 at end phase
DOTTE_CAP = 10          # Kirafan.Dotte10op


def is_battle_race(card):
    return card is not None and card.is_battle


def hp_of(c):
    """Current HP of a battle creamate (defense = number of HP cards)."""
    return c["def"]


def lethal_hp(c):
    """HP that has to be removed to defeat the card (나약함 makes 1 HP lethal)."""
    hp = hp_of(c)
    if c["counters"].get(COUNTER_WEAK, 0) > 0:
        return max(0, hp - 1)
    return hp


def compute(view, db):
    """Returns a dict of derived per-player numbers plus per-card flags keyed by id(card)."""
    out = {"players": [], "cards": {}}
    battle_by_rel = ([], [])
    for c in view["cards"]:
        card = db.get(c["code"]) if c["code"] else None
        c_is_battle = c["loc"] == LOCATION_MZONE and c["seq"] < 5 and is_battle_race(card)
        if c_is_battle:
            battle_by_rel[c["rel"]].append(c)

    for rel in (0, 1):
        p = view["players"][rel]
        field_level = 0
        hand_battle_levels = []
        hand_alchemists = 0
        for c in view["cards"]:
            if c["rel"] != rel:
                continue
            card = db.get(c["code"]) if c["code"] else None
            if c["loc"] in (LOCATION_MZONE, LOCATION_SZONE) and c["pos"] & POS_FACEUP:
                field_level += c["level"]
            if c["loc"] == LOCATION_HAND and card is not None:
                if card.is_battle:
                    hand_battle_levels.append(card.level)
                if card.class_index == 4:
                    hand_alchemists += 1
        starlight = p.get("summoner_def", 0)
        margin = starlight - field_level
        battle = battle_by_rel[rel]
        can_field_battle = len(battle) > 0 or any(lv <= margin for lv in hand_battle_levels)
        # Remaining attack power this turn: battle creamates with attack counters left. Before the battle
        # phase starts the counters are not placed yet, so every battle creamate counts as able to attack.
        attackers = []
        for c in battle:
            if c["counters"].get(COUNTER_ATTACKS, 0) > 0 or view["phase"] in (0, 1, 2, PHASE_MAIN1):
                attackers.append(c)
        out["players"].append({
            "sanctuary": p.get("sanctuary_level", 0),
            "to_realist": max(0, REALIST_LEVEL - p.get("sanctuary_level", 0)),
            "starlight": starlight,
            "field_level": field_level,
            "margin": margin,
            "battle_count": len(battle),
            "hand_battle": len(hand_battle_levels),
            "hand_alchemists": hand_alchemists,
            "can_field_battle": can_field_battle,
            "max_atk": max((c["atk"] for c in attackers), default=0),
            "sum_atk": sum(c["atk"] for c in attackers),
            "attack_counters": sum(c["counters"].get(COUNTER_ATTACKS, 0) for c in battle),
            "dotte": min(p.get("grave", 0), DOTTE_CAP),
            "hand": p.get("hand", 0),
            "hand_over": max(0, p.get("hand", 0) - HAND_CAP),
        })

    me, opp = out["players"]
    out["advantage"] = opp["sanctuary"] - me["sanctuary"]
    out["realist"] = max(me["sanctuary"], opp["sanctuary"]) >= REALIST_LEVEL
    out["mid"] = me["starlight"] >= MID_GAME_STARLIGHT
    out["my_turn"] = view["turn_player"] == 0
    out["in_battle"] = view["phase"] in (PHASE_BATTLE_START, PHASE_BATTLE_STEP, PHASE_DAMAGE,
                                         PHASE_DAMAGE_CAL, PHASE_BATTLE)

    # Per-card "can be defeated": by the strongest single attacker / by all attackers of the other side
    for rel in (0, 1):
        foe = out["players"][1 - rel]
        for c in battle_by_rel[rel]:
            need = lethal_hp(c)
            out["cards"][id(c)] = {
                "killable_single": foe["max_atk"] >= need > 0 or need == 0,
                "killable_total": foe["sum_atk"] >= need,
                "lethal_hp": need,
                "stiff": bool(c["pos"] & POS_DEFENSE),
                "bound": c["counters"].get(COUNTER_BIND, 0) > 0,
            }
    me["killable_levels"] = sum(c["level"] for c in battle_by_rel[0] if out["cards"][id(c)]["killable_total"])
    opp["killable_levels"] = sum(c["level"] for c in battle_by_rel[1] if out["cards"][id(c)]["killable_total"])
    return out


def evaluate(view, db, der=None):
    """State score from the player's point of view (guide section 6). Higher is better."""
    der = der or compute(view, db)
    me, opp = der["players"]
    s = der["advantage"] * 1000
    if not me["can_field_battle"]:
        s -= 100000
    late = der["realist"]
    mid = der["mid"]
    hand_w = 0 if late else (150 if not mid else 80)
    field_w = 60 if not mid else 120
    s += min(me["hand"], HAND_CAP) * hand_w
    s += me["field_level"] * field_w
    s += opp["killable_levels"] * 300
    s -= me["killable_levels"] * 300
    s -= max(0, me["dotte"] - DOTTE_CAP) * 50
    return s


def potential(view, db, der=None):
    """Shaping potential: sanctuary advantage scaled to roughly [-1, 1]."""
    der = der or compute(view, db)
    return der["advantage"] / 10.0


__all__ = ["compute", "evaluate", "potential", "REALIST_LEVEL", "MID_GAME_STARLIGHT", "HAND_CAP",
           "LOCATION_GRAVE", "LOCATION_EXTRA"]
