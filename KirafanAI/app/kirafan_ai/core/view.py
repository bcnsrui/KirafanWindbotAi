"""Player-visible game state ("view").

The same dict schema is produced by the simulator (from direct core queries, filtered to
what the player could see) and by the WindBot bridge (from the messages a real client
receives), so the model sees identical inputs in training and in EDOPro.

view = {
  "me": absolute player id, "turn": int, "turn_player": 0 (me) / 1 (opponent), "phase": int,
  "players": [summary for me, summary for opponent],
  "cards": [{"code", "rel", "loc", "seq", "pos", "atk", "def", "base_def", "level",
             "overlay", "counters", "public"}, ...],
  "chain": [{"code", "rel"}, ...],
  "attack": {"attacker": (rel, seq), "target": (rel, seq)} or None,
}
"""
from .constants import *  # noqa: F401,F403

PUBLIC_PILES = (LOCATION_GRAVE,)
# messages after which the attack being resolved is over (damage dealt, attack stopped, phase over)
ATTACK_END_MESSAGES = (MSG_ATTACK_DISABLED, MSG_BATTLE, MSG_DAMAGE_STEP_END, MSG_NEW_PHASE, MSG_NEW_TURN)


def targets_from_message(raw):
    """[(con, loc, seq)] of MSG_BECOME_TARGET (the cards an activation just targeted)."""
    n = int.from_bytes(raw[1:5], "little")
    return [(raw[5 + 10 * i], raw[6 + 10 * i], int.from_bytes(raw[7 + 10 * i:11 + 10 * i], "little")) for i in range(n)]


def attack_from_message(raw):
    """(attacker con, loc, seq, target con, loc, seq) of MSG_ATTACK."""
    return raw[1], raw[2], int.from_bytes(raw[3:7], "little"), raw[11], raw[12], int.from_bytes(raw[13:17], "little")


def make_card(code, rel, loc, seq, pos, atk=0, dfn=0, base_def=0, level=0, overlay=0, counters=None,
              public=1):
    return {"code": code, "rel": rel, "loc": loc, "seq": seq, "pos": pos, "atk": atk, "def": dfn,
            "base_def": base_def, "level": level, "overlay": overlay, "counters": counters or {},
            "public": public}


def summarize_players(view):
    """Fill derived per-player numbers (sanctuary level etc.) from the card list."""
    for rel in (0, 1):
        p = view["players"][rel]
        lvl = 0
        field_level = 0
        summoner = None
        # Stats a client is not told each time they change, worked out from the field: a battle creamate's
        # HP is its HP cards (Kirafan2.hpval); the summoner's attack is the level sum of its side's field
        # creamates (Kirafan.fieldlevel), its defense (성채석 한도) base + extra deck + its 성채석
        # (Kirafan.starlightstone)
        for c in view["cards"]:
            if c["rel"] != rel:
                continue
            if c["loc"] == LOCATION_EXTRA and c["pos"] & POS_FACEUP:
                lvl += c["level"]
            elif c["loc"] == LOCATION_MZONE and c["seq"] >= 5:
                summoner = c
            elif c["loc"] in (LOCATION_MZONE, LOCATION_SZONE) and c["pos"] & POS_FACEUP:
                field_level += c["level"]
                if c["loc"] == LOCATION_MZONE and c["code"]:
                    c["def"] = c["overlay"]
        if summoner is not None:
            summoner["atk"] = field_level
            summoner["def"] = summoner["base_def"] + p.get("extra", 0) + summoner["overlay"]
            p["summoner_atk"] = summoner["atk"]
            p["summoner_def"] = summoner["def"]
            p["summoner_overlay"] = summoner["overlay"]
            p["summoner_code"] = summoner["code"]
        p["sanctuary_level"] = lvl
        p.setdefault("summoner_atk", 0)
        p.setdefault("summoner_def", 0)
        p.setdefault("summoner_overlay", 0)
        p.setdefault("summoner_code", 0)
    return view


def attack_of(me, attack):
    """The attack being resolved (MSG_ATTACK until damage or the end of the attack), relative to `me`:
    {"attacker": (rel, seq), "target": (rel, seq)} for one battle zone creamate attacking another."""
    if not attack:
        return None
    a_con, a_loc, a_seq, t_con, t_loc, t_seq = attack
    if a_loc != LOCATION_MZONE or t_loc != LOCATION_MZONE:
        return None
    return {"attacker": (0 if a_con == me else 1, a_seq), "target": (0 if t_con == me else 1, t_seq)}


def view_from_queries(me, turn, turn_player, phase, field, locations, attack=None, chain_targets=None):
    """Build a view from core queries.

    field: parse_field() result; locations: {(player, loc): [card dict or None, ...]};
    attack: (attacker con, loc, seq, target con, loc, seq) of the attack being resolved, or None;
    chain_targets: per chain link [(con, loc, seq)] it targets (MSG_BECOME_TARGET), else in field["chain"].
    """
    view = {"me": me, "turn": turn, "turn_player": 0 if turn_player == me else 1, "phase": phase,
            "players": [], "cards": [], "chain": [], "attack": attack_of(me, attack)}
    for rel in (0, 1):
        absp = me if rel == 0 else 1 - me
        fp = field["players"][absp]
        view["players"].append({"lp": fp["lp"], "deck": fp["deck"], "hand": fp["hand"], "grave": fp["grave"],
                                "removed": fp["removed"], "extra": fp["extra"]})
        facedown_extra = 0
        for loc in (LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED,
                    LOCATION_EXTRA):
            for seq, q in enumerate(locations[(absp, loc)]):
                if q is None:
                    continue
                pos = q.get("pos", 0)
                public = q.get("public", 0)
                if loc == LOCATION_EXTRA and not (pos & POS_FACEUP):
                    facedown_extra += 1
                    if rel == 1:
                        continue
                if rel == 1 and loc == LOCATION_HAND and not public:
                    continue
                # 마을: the model only gets how many cards the opponent's holds (no card counting), face-up or not
                if rel == 1 and loc == LOCATION_REMOVED:
                    continue
                if rel == 1 and loc in (LOCATION_MZONE, LOCATION_SZONE) and not (pos & POS_FACEUP):
                    code = 0
                else:
                    code = q.get("code", 0)
                view["cards"].append(make_card(
                    code, rel, loc, seq, pos, q.get("atk", 0), q.get("def", 0), q.get("base_def", 0),
                    q.get("level", 0), len(q.get("overlay", ())), q.get("counters"), public))
        view["players"][rel]["extra_facedown"] = facedown_extra
    for i, ch in enumerate(field["chain"]):
        targets = chain_targets[i] if chain_targets is not None and i < len(chain_targets) else ch.get("targets", ())
        view["chain"].append({"code": ch["code"], "rel": 0 if ch["trig_con"] == me else 1,
                              "targets": [(0 if con == me else 1, loc, seq) for con, loc, seq in targets]})
    return summarize_players(view)
