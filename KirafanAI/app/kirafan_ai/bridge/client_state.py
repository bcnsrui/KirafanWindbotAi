"""The game state one player can rebuild from the messages EDOPro sends it.

EDOPro's location refreshes carry codes, positions and stats but no counters and no overlays,
so those are tracked from MSG_ADD/REMOVE_COUNTER and MSG_MOVE. view() returns the same dict as
sim.duel.Duel.view(), built with core.view.view_from_queries.
"""
import struct

from ..cards import ATTRIBUTE_NAMES
from ..cardtext import text_of
from ..core.constants import *  # noqa: F401,F403
from ..core.query import parse_card
from ..core.reader import Reader
from ..core.view import view_from_queries, attack_from_message, targets_from_message, ATTACK_END_MESSAGES

ZONE_SLOTS = {LOCATION_MZONE: 7, LOCATION_SZONE: 8}
PILES = (LOCATION_DECK, LOCATION_HAND, LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA)
FIELD = LOCATION_MZONE | LOCATION_SZONE


def parse_cards(raw, pos):
    """A location refresh: [u32 size] then cards whose private fields may be missing."""
    r = Reader(raw, pos)
    end = r.pos + 4 + r.u32()
    cards = []
    while r.pos < end:
        cards.append(parse_card(r))
    return cards


def new_card(code=0, pos=POS_FACEDOWN_DEFENSE):
    return {"code": code, "pos": pos, "overlay": [], "counters": {}}


class ClientState:
    def __init__(self, db):
        self.db = db
        self.me = 0
        self.started = False
        self.lp = [0, 0]
        self.turn = 0
        self.turn_player = 0
        self.phase = 0
        self.hints = [0, 0]
        self.chain = []
        self.waiting = 0          # MSG_WAITING count: the opponent was asked to decide
        self.attack = None
        self.winner = None
        self.locs = {}
        for p in (0, 1):
            for loc, n in ZONE_SLOTS.items():
                self.locs[(p, loc)] = [None] * n
            for loc in PILES:
                self.locs[(p, loc)] = []

    # ---- helpers ------------------------------------------------------
    def _get(self, con, loc, seq):
        cards = self.locs.get((con, loc))
        if cards is None or not 0 <= seq < len(cards):
            return None
        return cards[seq]

    def _take(self, con, loc, seq, pos):
        """Remove and return the card at a location (a fresh card if unknown)."""
        if loc & LOCATION_OVERLAY:
            host = self._get(con, loc & 0x7f, seq)
            if host and 0 <= pos < len(host["overlay"]):
                return new_card(host["overlay"].pop(pos), POS_FACEUP_ATTACK)
            return new_card()
        cards = self.locs.get((con, loc))
        if cards is None:
            return new_card()
        if loc in ZONE_SLOTS:
            card = cards[seq] if 0 <= seq < len(cards) else None
            if 0 <= seq < len(cards):
                cards[seq] = None
            return card or new_card()
        if 0 <= seq < len(cards):
            return cards.pop(seq)
        return cards.pop() if cards else new_card()

    def _put(self, card, con, loc, seq, pos):
        if loc == 0:
            return
        if loc & LOCATION_OVERLAY:
            host = self._get(con, loc & 0x7f, seq)
            if host is not None:
                host["overlay"].insert(min(pos, len(host["overlay"])), card["code"])
            return
        card["pos"] = pos
        cards = self.locs[(con, loc)]
        if loc in ZONE_SLOTS:
            if 0 <= seq < len(cards):
                cards[seq] = card
            return
        cards.insert(min(seq, len(cards)), card)

    def _merge(self, old, new):
        """A refreshed card keeps what refreshes never carry (counters, overlays)."""
        if new is None:
            return None
        card = new_card()
        card.update(new)
        if old is not None:
            card["counters"] = old["counters"]
            card["overlay"] = old["overlay"]
        return card

    # ---- stats that follow the board -----------------------------------
    # EDOPro refreshes a card's stats only at some messages, while board-dependent "[패시브]" attack / level
    # (cardtext passive_atk) and 수면 change with every move. The part they made of the stats is noted at each
    # refresh ("board_at") and brought up to date in view(); what else the refresh carried (effect buffs) stays.
    def _field_creamates(self, p):
        return [c for loc in (LOCATION_MZONE, LOCATION_SZONE) for c in self.locs[(p, loc)][:5]
                if c is not None and c["code"] and c.get("pos", POS_FACEUP) & POS_FACEUP]

    def _sanctuary(self, p):
        return sum(self.db[c["code"]].level for c in self.locs[(p, LOCATION_EXTRA)]
                   if c["code"] and c.get("pos", 0) & POS_FACEUP and self.db.get(c["code"]))

    def _attr(self, c):
        card = self.db.get(c["code"])
        return card.attribute if card is not None else 0

    def _board_part(self, p, card, on_field):
        """(attack, level) the board adds to this card of player p now: its passives and, on the field, 수면."""
        atk = lvl = 0
        mine = self._field_creamates(p)
        for rule in text_of(self.db, card["code"]).passive_atk:
            name = rule[0]
            if name == "with_card":
                if any(self.db.name(c["code"]) == rule[1] for c in mine):
                    lvl -= int(rule[2])
                    atk += int(rule[3])
                continue
            if not on_field:
                continue
            hp = len(card["overlay"])
            if name == "field_ge" and len(mine) >= int(rule[1]):
                atk += int(rule[2])
            elif name == "attr_ge" and sum(1 for c in mine if ATTRIBUTE_NAMES.get(self._attr(c)) == rule[1]) >= int(rule[2]):
                atk += int(rule[3])
            elif name == "hp_le" and hp <= int(rule[1]):
                atk -= int(rule[2])
            elif name == "hp_ge" and hp >= int(rule[1]):
                atk += int(rule[2])
            elif name == "no_support" and not any(c for c in self.locs[(p, LOCATION_SZONE)][:5] if c and c["code"]):
                atk += int(rule[1])
            elif name == "sanct_higher" and self._sanctuary(p) > self._sanctuary(1 - p):
                atk += int(rule[1])
            elif name == "professor":
                attrs = {self._attr(c) for c in mine}
                atk += sum(1 for c in self._field_creamates(1 - p) if self._attr(c) in attrs)
        if on_field and card["counters"].get(COUNTER_SLEEP, 0) > 0:
            atk -= 1
        return atk, lvl

    def _note_board(self, p, loc):
        for c in self.locs[(p, loc)]:
            if c is not None and c["code"] and ("atk" in c or "level" in c):
                c["board_at"] = self._board_part(p, c, loc == LOCATION_MZONE)

    def _board_view(self):
        """self.locs with the field and hand stats brought up to the board now (copies of the changed cards)."""
        locs = dict(self.locs)
        for p in (0, 1):
            for loc in (LOCATION_MZONE, LOCATION_HAND):
                cards = list(locs[(p, loc)])
                for i, c in enumerate(cards):
                    if c is None or not c["code"] or (loc == LOCATION_MZONE and i >= 5):
                        continue
                    was = c.get("board_at", (0, 0))   # (printed stats from _fill_stats carry none)
                    now = self._board_part(p, c, loc == LOCATION_MZONE)
                    c2 = dict(c)
                    if loc == LOCATION_MZONE and c["counters"].get(COUNTER_ISOLATE, 0) > 0:
                        card = self.db.get(c["code"])
                        c2["atk"] = c.get("base_atk", card.attack if card else 0)   # (고립: Kirafan2.alonevalue)
                    elif "atk" in c:
                        c2["atk"] = c["atk"] - was[0] + now[0]
                    if "level" in c:
                        c2["level"] = c["level"] - was[1] + now[1]
                    cards[i] = c2
                locs[(p, loc)] = cards
        return locs

    # ---- messages -----------------------------------------------------
    def feed(self, raw):
        msg = raw[0]
        if msg == MSG_ATTACK:
            self.attack = attack_from_message(raw)
        elif msg in ATTACK_END_MESSAGES:
            self.attack = None
        handler = HANDLERS.get(msg)
        if handler:
            handler(self, raw)

    def on_start(self, raw):
        r = Reader(raw, 1)
        self.me = r.u8() & 0xf
        self.lp = [r.u32(), r.u32()]
        for p in (0, 1):
            deck, extra = r.u16(), r.u16()
            self.locs[(p, LOCATION_DECK)] = [new_card() for _ in range(deck)]
            self.locs[(p, LOCATION_EXTRA)] = [new_card() for _ in range(extra)]
        self.started = True

    def on_update_data(self, raw):
        player, loc = raw[1], raw[2]
        if (player, loc) not in self.locs:
            return
        new = parse_cards(raw, 3)
        old = self.locs[(player, loc)]
        if loc in ZONE_SLOTS:
            n = ZONE_SLOTS[loc]
            new = (new + [None] * n)[:n]
            self.locs[(player, loc)] = [self._merge(old[i], new[i]) for i in range(n)]
        else:
            self.locs[(player, loc)] = [self._merge(old[i] if i < len(old) else None, c) or new_card()
                                        for i, c in enumerate(new)]
        if loc in (LOCATION_MZONE, LOCATION_HAND):
            self._note_board(player, loc)

    def on_update_card(self, raw):
        player, loc, seq = raw[1], raw[2], raw[3]
        cards = self.locs.get((player, loc))
        if cards is None or not 0 <= seq < len(cards):
            return
        info = parse_card(Reader(raw, 4))
        if info is None:
            return
        if cards[seq] is None:
            cards[seq] = new_card()
        cards[seq].update(info)
        if loc in (LOCATION_MZONE, LOCATION_HAND) and cards[seq]["code"]:
            cards[seq]["board_at"] = self._board_part(player, cards[seq], loc == LOCATION_MZONE)

    def on_move(self, raw):
        r = Reader(raw, 1)
        code = r.u32()
        pcon, ploc, pseq, ppos = r.loc_info()
        ccon, cloc, cseq, cpos = r.loc_info()
        card = self._take(pcon, ploc, pseq, ppos)
        card["code"] = code
        if (ploc & FIELD) and not (ploc & LOCATION_OVERLAY) and not (cloc & FIELD and cloc == ploc and ccon == pcon):
            card["counters"] = {}
        if not (cloc & FIELD) or cloc & LOCATION_OVERLAY:
            card["overlay"] = []
        for k in ("atk", "def", "base_def", "level", "public", "board_at"):
            if not (cloc & FIELD):
                card.pop(k, None)
        self._put(card, ccon, cloc, cseq, cpos)

    def on_draw(self, raw):
        r = Reader(raw, 1)
        player = r.u8()
        deck = self.locs[(player, LOCATION_DECK)]
        hand = self.locs[(player, LOCATION_HAND)]
        for _ in range(r.u32()):
            code, pos = r.u32(), r.u32()
            card = deck.pop() if deck else new_card()
            card["code"] = code
            card["pos"] = pos
            hand.append(card)

    def on_swap(self, raw):
        r = Reader(raw, 1)
        c1 = r.u32()
        a = r.loc_info()
        c2 = r.u32()
        b = r.loc_info()
        x = self._get(a[0], a[1], a[2])
        y = self._get(b[0], b[1], b[2])
        if x is not None and y is not None:
            self.locs[(a[0], a[1])][a[2]], self.locs[(b[0], b[1])][b[2]] = y, x
            y["code"], x["code"] = c1 or y["code"], c2 or x["code"]
            y["pos"], x["pos"] = a[3], b[3]

    def on_pos_change(self, raw):
        code = struct.unpack_from("<I", raw, 1)[0]
        con, loc, seq, cur = raw[5], raw[6], raw[7], raw[9]
        card = self._get(con, loc, seq)
        if card is not None:
            card["pos"] = cur
            if code:
                card["code"] = code

    def on_shuffle_hand(self, raw):
        player = raw[1]
        count = struct.unpack_from("<I", raw, 2)[0]
        hand = self.locs[(player, LOCATION_HAND)]
        codes = struct.unpack_from("<%dI" % count, raw, 6)
        for card, code in zip(hand, codes):
            card["code"] = code

    def on_lp(self, raw):
        player = raw[1]
        v = struct.unpack_from("<I", raw, 2)[0]
        if player > 1:
            return
        if raw[0] in (MSG_DAMAGE, MSG_PAY_LPCOST):
            self.lp[player] = max(0, self.lp[player] - v)
        elif raw[0] == MSG_RECOVER:
            self.lp[player] += v
        else:
            self.lp[player] = v

    def on_counter(self, raw):
        ctype = struct.unpack_from("<H", raw, 1)[0]
        con, loc, seq = raw[3], raw[4], raw[5]
        # (signed: a status re-applied with fewer turns than it has left adds a negative count, Kirafan6.statuseff)
        count = struct.unpack_from("<h", raw, 6)[0]
        card = self._get(con, loc, seq)
        if card is None:
            return
        n = card["counters"].get(ctype, 0) + (count if raw[0] == MSG_ADD_COUNTER else -count)
        if n > 0:
            card["counters"][ctype] = n
        else:
            card["counters"].pop(ctype, None)

    def on_chaining(self, raw):
        r = Reader(raw, 1)
        code = r.u32()
        r.loc_info()
        trig_con = r.u8()
        self.chain.append({"code": code, "trig_con": trig_con, "targets": []})

    def on_become_target(self, raw):
        if self.chain:
            self.chain[-1]["targets"] = targets_from_message(raw)

    def on_chain_solved(self, raw):
        if self.chain:
            self.chain.pop()

    def on_chain_end(self, raw):
        self.chain = []

    def on_new_turn(self, raw):
        self.turn += 1
        self.turn_player = raw[1]

    def on_new_phase(self, raw):
        self.phase = struct.unpack_from("<H", raw, 1)[0]

    def on_hint(self, raw):
        typ, player = raw[1], raw[2]
        if typ == HINT_SELECTMSG and player < 2:
            self.hints[player] = struct.unpack_from("<Q", raw, 3)[0]

    def on_waiting(self, raw):
        self.waiting += 1

    def on_win(self, raw):
        self.winner = raw[1]

    # ---- view ---------------------------------------------------------
    def field(self):
        players = []
        for p in (0, 1):
            players.append({"lp": self.lp[p], "deck": len(self.locs[(p, LOCATION_DECK)]),
                            "hand": len(self.locs[(p, LOCATION_HAND)]),
                            "grave": len(self.locs[(p, LOCATION_GRAVE)]),
                            "removed": len(self.locs[(p, LOCATION_REMOVED)]),
                            "extra": len(self.locs[(p, LOCATION_EXTRA)])})
        return {"players": players, "chain": [dict(c) for c in self.chain]}

    def locations(self):
        return self.locs

    def _fill_stats(self):
        """Cards off the field that no refresh described yet carry their printed stats; so does the HP
        of a battle creamate that was summoned mid-chain (EDOPro refreshes it only later, and a battle
        creamate with 0 HP cannot be on the field)."""
        for p in (0, 1):
            for c in self.locs[(p, LOCATION_MZONE)][:5]:
                if c is None or not c["code"] or c.get("def", 0) > 0:
                    continue
                card = self.db.get(c["code"])
                if card is not None and card.defense > 0:
                    c["base_def"] = c.get("base_def") or card.defense
                    c["def"] = c["base_def"]
        for (p, loc), cards in self.locs.items():
            if loc in ZONE_SLOTS or loc == LOCATION_DECK:
                continue
            for c in cards:
                if "atk" in c or not c["code"]:
                    continue
                card = self.db.get(c["code"])
                if card is not None:
                    c.update(atk=card.attack, base_atk=card.attack, level=card.level)
                    c["def"] = c["base_def"] = card.defense

    def view(self, player=None):
        self._fill_stats()
        return view_from_queries(self.me if player is None else player, self.turn, self.turn_player, self.phase,
                                 self.field(), self._board_view(), self.attack)


HANDLERS = {
    MSG_START: ClientState.on_start, MSG_UPDATE_DATA: ClientState.on_update_data,
    MSG_UPDATE_CARD: ClientState.on_update_card, MSG_MOVE: ClientState.on_move, MSG_DRAW: ClientState.on_draw,
    MSG_SWAP: ClientState.on_swap, MSG_POS_CHANGE: ClientState.on_pos_change,
    MSG_SHUFFLE_HAND: ClientState.on_shuffle_hand,
    MSG_DAMAGE: ClientState.on_lp, MSG_RECOVER: ClientState.on_lp, MSG_LPUPDATE: ClientState.on_lp,
    MSG_PAY_LPCOST: ClientState.on_lp,
    MSG_ADD_COUNTER: ClientState.on_counter, MSG_REMOVE_COUNTER: ClientState.on_counter,
    MSG_CHAINING: ClientState.on_chaining, MSG_CHAIN_SOLVED: ClientState.on_chain_solved,
    MSG_BECOME_TARGET: ClientState.on_become_target,
    MSG_CHAIN_END: ClientState.on_chain_end, MSG_NEW_TURN: ClientState.on_new_turn,
    MSG_NEW_PHASE: ClientState.on_new_phase, MSG_HINT: ClientState.on_hint, MSG_WAITING: ClientState.on_waiting,
    MSG_WIN: ClientState.on_win,
}
