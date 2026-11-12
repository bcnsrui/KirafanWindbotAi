"""The process WindBot starts for a KirafanAI bot.

    python -m kirafan_ai.bridge.server --model WindBot/KirafanAI/models/18400 [--level 1-4] [--mode best|sample]

The model is <name>.npz (numpy, the portable bundle) or <name>.pt (PyTorch checkpoint).
stdin:  [u32 length][game message] for every STOC_GAME_MSG the bot receives
stdout: [u32 length][response bytes] for every message (length 0: nothing to answer)
"""
import argparse
import os
import random
import struct
import sys
import time
import traceback

from ..cards import get_db
from ..core.constants import LOCATION_HAND
from ..core.messages import KIND_NAMES, auto_response, parse_decision
from ..model.features import encode, pack
from .session import Session


def option_text(db, o):
    t = KIND_NAMES.get(o.kind, str(o.kind))
    if o.card and o.card[0]:
        t += " " + db.name(o.card[0])
    if o.desc and (o.desc >> 20):
        t += " [%s]" % db.describe(o.desc)
    return t


def softmax(x):
    import numpy as np
    e = np.exp(x - x.max())
    return e / e.sum()


BLUFF_HAND = 5         # with this many cards in our hand, one heal card spent on a bluffed save costs little
BLUFF_MAX_P = 0.5      # ... else go for it while the opponent holds an attack buff at most this likely
BLUFF_MIN_PROB = 0.05  # (only a move the model itself gives some chance)
DECK_SIZE, DECK_BUFFS = 60, 6   # assumed attack buffs in the opponent's 60-card deck (its list is hidden)


def p_holds_buff(hand):
    """Chance that `hand` cards drawn from a 60-card deck with DECK_BUFFS buffs hold at least one."""
    p_none = 1.0
    for i in range(int(hand)):
        p_none *= (DECK_SIZE - DECK_BUFFS - i) / (DECK_SIZE - i)
    return 1.0 - p_none


def bluff_heal(view, enc, probs, rng):
    """The mind game when defending: a heal that saves our creamate this turn unless the opponent has an
    attack buff in its hidden hand (option feature 35). Go for it when one card is cheap for us or the buff
    is not that likely; otherwise play the model's own odds (never always the same, so it cannot be read).
    Returns (option or None, note)."""
    from ..model.features import OPTION_BASE_DIM
    cands = [i for i, row in enumerate(enc["opts"]) if row[OPTION_BASE_DIM + 35] > 0 and probs[i] >= BLUFF_MIN_PROB]
    if not cands:
        return None, ""
    mine, opp = view["players"][0].get("hand", 0), view["players"][1].get("hand", 0)
    if mine >= BLUFF_HAND or p_holds_buff(opp) <= BLUFF_MAX_P:
        return max(cands, key=lambda i: probs[i]), " | 심리전: 손해가 적어 숨은 버프가 없으면 살리는 회복을 지름"
    return rng.choices(range(len(probs)), weights=[float(p) for p in probs])[0], " | 심리전: 모델 확률대로 섞음"


PLAN_TRUST = 0.5       # below this probability for its own first choice the model is unsure ...
PLAN_MIN_PROB = 0.03   # ... and the look-ahead's first step (a move the model gives some chance) is played
#                        when the look-ahead defeats more enemies this turn with it


def plan_step(enc, probs, idx):
    """The look-ahead's best next step (option features 36-41, model.tactical) instead of option idx, or
    None."""
    from ..model.features import OPTION_BASE_DIM as B
    if probs[idx] >= PLAN_TRUST:
        return None
    rows = enc["opts"]
    mine = rows[idx][B + 39] if rows[idx][B + 41] > 0 else None
    cands = [i for i, r in enumerate(rows) if r[B + 36] > 0 and r[B + 41] > 0 and probs[i] >= PLAN_MIN_PROB]
    if not cands:
        return None
    best = max(cands, key=lambda i: (rows[i][B + 39], probs[i]))
    if rows[best][B + 39] <= 0 or (mine is not None and rows[best][B + 39] <= mine):
        return None   # (only for a plan that defeats something the model's own move does not)
    return best


def defense_step(enc, probs, idx):
    """The defense look-ahead's best answer to the attack (option features 42-47, model.tactical.Defense)
    instead of option idx when the model is unsure and that answer loses fewer of our creamates, or None."""
    from ..model.features import OPTION_BASE_DIM as B
    if probs[idx] >= PLAN_TRUST:
        return None
    rows = enc["opts"]
    mine = rows[idx][B + 43] if rows[idx][B + 47] > 0 else None
    cands = [i for i, r in enumerate(rows) if r[B + 42] > 0 and r[B + 47] > 0 and probs[i] >= PLAN_MIN_PROB]
    if not cands:
        return None
    best = min(cands, key=lambda i: (rows[i][B + 43], -probs[i]))
    # better than the model's own answer, or (when its move is none the look-ahead knows) than passing
    if rows[best][B + 43] >= (mine if mine is not None else rows[best][B + 44]):
        return None
    return best


def option_groups(d):
    """Per option a key shared by the options that are the same move (copies of one card in the hand)."""
    return [(o.kind, o.card[0], o.card[1], o.desc) if o.card and o.card[2] == LOCATION_HAND else i
            for i, o in enumerate(d.options)]


def best_of(groups, probs):
    """The most likely move, counting copies of one card in the hand as one move: two copies of a card
    split its probability between two options, so a plain argmax would pass it over."""
    best = {}
    for i, key in enumerate(groups):
        g = best.setdefault(key, [0.0, i])
        g[0] += float(probs[i])
    return max(best.values(), key=lambda g: g[0])[1]


def best_option(d, probs):
    return best_of(option_groups(d), probs)


# Difficulty levels (bots.json difficulty; the bot's deck "KirafanAI:<model>@<level>"): None plays the best move
# with the look-ahead and mind-game overrides; else (temperature, least probability a move needs) of the
# model's own odds it draws from - a flatter draw errs more, most where the model is unsure. Tuned with
# kirafan_ai.eval.levels (model 12280, ~96 games a pair): 4 beats 3 61%, 3 beats 2 70%, 2 beats 1 75%
LEVELS = {4: None, 3: (2.2, 0.0), 2: (3.5, 0.0), 1: (5.5, 0.0)}


def drawn(probs, temperature, floor, rng):
    w = [float(p) ** (1.0 / temperature) if float(p) >= floor else 0.0 for p in probs]
    if not any(w):
        w = [float(p) for p in probs]
    return rng.choices(range(len(probs)), weights=w)[0]


def choose(level, groups, rows, probs, hands, rng):
    """(option, note) the bot of this level plays: groups (option_groups), rows: the encoded options
    (enc["opts"]), hands: (our hand, the opponent's hand) for the mind game."""
    from ..model.features import OPTION_BASE_DIM
    params = LEVELS.get(level)
    if params is not None:
        return drawn(probs, params[0], params[1], rng), ""
    idx, note = best_of(groups, probs), ""
    enc = {"opts": rows}
    if rows[idx][OPTION_BASE_DIM + 35] <= 0:   # (a heal that is already first is played as it is)
        bluff, why = bluff_heal({"players": [{"hand": hands[0]}, {"hand": hands[1]}]}, enc, probs, rng)
        if bluff is not None:
            idx, note = bluff, why
    step = plan_step(enc, probs, idx) if not note else None
    if step is not None and step != idx:
        idx, note = step, " | 수읽기: 이번 턴에 더 많이 쓰러뜨리는 순서 (%.0f체)" % (rows[step][OPTION_BASE_DIM + 39] * 3)
    guard = defense_step(enc, probs, idx) if not note else None
    if guard is not None and guard != idx:
        idx, note = guard, " | 수읽기(방어): 이번 턴에 아군을 덜 잃는 대응 (%.0f체 잃음)" % (rows[guard][OPTION_BASE_DIM + 43] * 3)
    return idx, note


def load(path):
    """Returns (db, update, forward) where forward(packed observation) -> (probs, value)."""
    if not os.path.splitext(path)[1]:
        path = path + ".npz" if os.path.exists(path + ".npz") else path + ".pt"
    db = get_db()
    if path.endswith(".npz"):
        from .npmodel import NumpyPolicy, to_arrays
        net = NumpyPolicy(path)
        db.set_vocab_order(net.vocab)

        def forward(enc):
            logits, value = net.forward(to_arrays(enc))
            return softmax(logits), value
        return db, net.update, forward
    import torch
    from ..model.net import build, collate, fit_state_dict, to_arrays
    ck = torch.load(path, map_location="cpu", weights_only=False)
    db.set_vocab_order(ck["vocab"])
    model = build(len(db.vocab_codes))
    model.load_state_dict(fit_state_dict(ck["model"], model))
    model.eval()

    def forward(enc):
        with torch.inference_mode():
            logits, value = model(collate([to_arrays(enc)], "cpu"))
        return torch.softmax(logits[0], -1).numpy(), float(value[0])
    return db, ck.get("update"), forward


# The bot's lines during the game: from its chat file (Dialogs/KirafanAI_S<summoner>.json for its deck's
# summoner card - the one WindBot greets with, KirafanBridge.DialogName), key ->
# chance of saying one of that key's lines ("{0}": the card) when it comes up; under CHAT_VS["<the opponent's
# summoner>"] the file may give lines of any key for that opponent, said instead of its own ones
CHAT_CHANCE = {"내턴": 0.25, "소환": 0.35, "공격": 0.2, "힐": 0.5, "돗테오키": 0.5, "서포트": 0.3,
               "킬": 0.7, "쓰러짐": 0.5, "승리": 1.0, "패배": 1.0}
CHAT_GAP = 3            # decisions at least between two lines (but for these)
CHAT_ALWAYS = ("승리", "패배", "진실의손", "만남")   # ("만남": once the opponent's summoner is known)
CHAT_VS = "상대"
# lines of keys added after chat files were written (a file without the key says these)
CHAT_DEFAULTS = {"진실의손": ["진실의 손의 시간이에요!"]}
# Kirafan.realistop's hint to its owner as its 진실의 손의 시간 starts (the opponent's / its own sanctuary at 10)
REALIST_HINTS = ((10050114 << 20) | 4, (10050114 << 20) | 5)
HINT_MESSAGE = 2


def chat_file(summoner):
    """Path of the bot's chat file, or None."""
    from .. import paths
    root = os.path.join(paths.GAME_ROOT, "WindBot", "Dialogs")
    path = os.path.join(root, "KirafanAI_S%d.json" % summoner)
    return path if summoner and os.path.exists(path) else None


class Chatter:
    """Lines the bot says, sent to WindBot ahead of its answer to a message (KirafanBridge.Handle)."""

    def __init__(self, db, level, rng):
        self.db, self.level, self.rng = db, level, rng
        self.lines = None     # key -> [line], once the summoner is known
        self.vs = {}          # the lines for the opponent's summoner, once it is known
        self.opponent = 0
        self.quiet = 0
        self.out = []

    def load(self, summoner, opponent=0):
        if self.lines is None:
            import json
            path = chat_file(summoner)
            try:
                with open(path, encoding="utf-8-sig") as f:
                    self.lines = json.load(f) if path else {}
            except (OSError, ValueError, TypeError):
                self.lines = {}
            if not isinstance(self.lines, dict):
                self.lines = {}
        if opponent and not self.opponent:
            self.opponent = opponent
            vs = self.lines.get(CHAT_VS)
            if isinstance(vs, dict):
                found = vs.get(self.card(opponent).strip(), vs.get(str(opponent)))
                self.vs = found if isinstance(found, dict) else {}
            self.say("만남", opponent)

    def card(self, code):
        return self.db.name(code).replace("(소환사 카드)", "") if code else ""

    @staticmethod
    def _lines(found):
        return [x for x in found or () if isinstance(x, str) and x] if isinstance(found, list) else []

    def say(self, key, code=0):
        if self.lines is None:
            return
        lines = self._lines(self.vs.get(key)) or self._lines(self.lines.get(key, CHAT_DEFAULTS.get(key)))
        if not lines:
            return
        if key not in CHAT_ALWAYS and (self.quiet > 0 or self.rng.random() >= CHAT_CHANCE.get(key, 0.0)):
            return
        self.out.append(self.rng.choice(lines).replace("{0}", self.card(code)))
        self.quiet = CHAT_GAP

    def decided(self, d, o):
        """A line for the move this bot just chose: its heal, dotte, attack or support."""
        from ..core.constants import LOCATION_MZONE
        from ..core.messages import ACTIVATE_KINDS, K_BATTLE_ATTACK
        from ..model.features import SINGLE_HEAL_DESC, ALL_HEAL_DESC, DOTTE_DESCS
        from ..cardtext import text_of
        self.quiet -= 1
        if o.card is None:
            return
        code, loc = o.card[0], o.card[2]
        if o.kind == K_BATTLE_ATTACK:
            self.say("공격", code)
        elif o.kind in ACTIVATE_KINDS:
            card = self.db.get(code)
            t = text_of(self.db, code)
            if o.desc in (SINGLE_HEAL_DESC, ALL_HEAL_DESC) or \
                    (o.desc in DOTTE_DESCS and (t.dotte_heal_single or t.dotte_heal_all)):
                self.say("힐", code)
            elif loc == LOCATION_MZONE and o.desc in DOTTE_DESCS and card is not None and card.is_battle:
                self.say("돗테오키", code)
            elif card is not None and card.is_support:
                self.say("서포트", code)

    def message(self, raw, me):
        """A line for what just happened: our turn, a summon of ours, a creamate defeated, our 진실의 손의 시간,
        the end."""
        from ..core.constants import (MSG_NEW_TURN, MSG_MOVE, MSG_WIN, MSG_HINT, LOCATION_MZONE, LOCATION_EXTRA,
                                      LOCATION_HAND)
        msg = raw[0]
        if msg == MSG_NEW_TURN and raw[1] == me:
            self.say("내턴")
        elif msg == MSG_HINT and raw[1] == HINT_MESSAGE and raw[2] == me \
                and struct.unpack_from("<Q", raw, 3)[0] in REALIST_HINTS:
            self.say("진실의손")
        elif msg == MSG_WIN and raw[1] < 2:
            self.say("승리" if raw[1] == me else "패배")
        elif msg == MSG_MOVE:
            code = struct.unpack_from("<I", raw, 1)[0]
            card = self.db.get(code)
            if card is None or not card.is_battle:
                return
            fcon, floc, fseq = raw[5], raw[6], struct.unpack_from("<I", raw, 7)[0]
            tcon, tloc, tseq = raw[15], raw[16], struct.unpack_from("<I", raw, 17)[0]
            if floc == LOCATION_MZONE and fseq < 5 and tloc == LOCATION_EXTRA:
                self.say("쓰러짐" if fcon == me else "킬", code)
            elif floc == LOCATION_HAND and tloc == LOCATION_MZONE and tseq < 5 and tcon == me:
                self.say("소환", code)

    def flush(self):
        """The chat records to write before the answer (length -1, [u32 size][UTF-8])."""
        data = b"".join(struct.pack("<iI", -1, len(t.encode("utf-8"))) + t.encode("utf-8") for t in self.out)
        self.out = []
        return data


def read_exact(f, n):
    data = b""
    while len(data) < n:
        chunk = f.read(n - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="best", choices=("best", "sample"))
    ap.add_argument("--level", type=int, default=0, help="difficulty 1-4 (LEVELS); 0: --mode")
    ap.add_argument("--log")
    args = ap.parse_args()
    level = args.level if args.level in LEVELS else (4 if args.mode == "best" else 3)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    log_file = open(args.log, "a", encoding="utf-8") if args.log else None

    def log(text):
        if log_file:
            log_file.write(text + "\n")
            log_file.flush()

    db, update, forward = load(args.model)
    rng = random.Random()
    log("=== %s | 모델 %s (업데이트 %s) | 난이도 %d단계 (%s)" % (
        time.strftime("%Y-%m-%d %H:%M:%S"), os.path.basename(args.model), update, level,
        "가장 좋은 수" if LEVELS[level] is None else "확률 선택, 온도 %.1f" % LEVELS[level][0]))

    chatter = Chatter(db, level, rng)

    def policy(view, d):
        enc = encode(view, d, db)
        probs, value = forward(pack(enc))
        idx, note = choose(level, option_groups(d), enc["opts"], probs,
                           (view["players"][0].get("hand", 0), view["players"][1].get("hand", 0)), rng)
        chatter.load(view["players"][0].get("summoner_code", 0), view["players"][1].get("summoner_code", 0))
        chatter.decided(d, d.options[idx])
        log("  T%d %s (%.0f%%) | 예상 승률 %.0f%%%s" % (view["turn"], option_text(db, d.options[idx]),
                                                    100 * float(probs[idx]), 50 * (value + 1), note))
        return idx

    session = Session(db, policy, log=log)
    while True:
        try:
            n = struct.unpack("<I", read_exact(stdin, 4))[0]
            raw = read_exact(stdin, n)
        except EOFError:
            break
        try:
            resp = session.on_message(raw)
        except Exception:
            log("오류:\n" + traceback.format_exc())
            d = parse_decision(raw) if raw[0] != 5 else None
            resp = None
            if d is not None and raw[1] == session.state.me:
                resp = auto_response(d) if d.meta.get("auto") else session._fallback(d)
        if raw[0] == 5:   # MSG_WIN
            log("  결과: %s" % ("승리" if raw[1] == session.state.me else "패배" if raw[1] < 2 else "무승부"))
        try:
            chatter.message(raw, session.state.me)
        except Exception:
            log("대사 오류:\n" + traceback.format_exc())
        for line in chatter.out:
            log("  (대사) " + line)
        stdout.write(chatter.flush() + struct.pack("<I", len(resp or b"")) + (resp or b""))
        stdout.flush()


if __name__ == "__main__":
    main()
