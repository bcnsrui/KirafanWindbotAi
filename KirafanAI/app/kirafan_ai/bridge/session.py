"""Turns the messages one player receives into responses, with the same decision flow as sim.duel.

The policy only sees what training showed it: the ClientState view, the parsed Decision with
its context (last activation / attack / summon), static and dynamic no-ops, and multi-card
selections split into single picks. Decisions the policy never controls get auto_response.
"""
import random

from ..core.constants import *  # noqa: F401,F403
from ..core.messages import (parse_decision, auto_response, encode_response, POLICY_MESSAGES, PROGRESS_KINDS,
                             CONTEXT_KINDS, Option)
from .client_state import ClientState


class Session:
    def __init__(self, db, policy, log=None, rng=None):
        """policy(view, decision) -> option index."""
        self.db = db
        self.policy = policy
        self.log = log or (lambda *a: None)
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        self.state = ClientState(self.db)
        self.last_action = None
        self._visits = {}
        self._noops = {}

    def on_message(self, raw):
        """Feed one game message; returns response bytes when this player must answer, else None."""
        if raw[0] == MSG_START:
            self.reset()
        self.state.feed(raw)
        if raw[0] not in DECISION_MESSAGES or raw[1] != self.state.me:
            return None
        d = parse_decision(raw, self.state.hints[raw[1]])
        if d is None:
            return None
        self._fill_codes(d)
        if d.meta.get("auto"):
            return auto_response(d, self.rng)
        if d.msg not in POLICY_MESSAGES or not d.options:
            self.log("처리하지 못한 결정 %d (선택지 %d개): 기본 응답" % (d.msg, len(d.options)))
            return self._fallback(d)
        if d.msg in (MSG_SELECT_IDLECMD, MSG_SELECT_BATTLECMD, MSG_SELECT_CHAIN):
            self.last_action = None
        while True:
            d.context = self.last_action
            idx = 0 if len(d.options) == 1 else self._choose(d)
            opt = d.options[idx]
            if opt.kind in CONTEXT_KINDS:
                self.last_action = (opt.kind, opt.card, opt.desc)
            data = d.selection.apply(opt) if d.selection is not None else encode_response(d, opt)
            if data is not None:
                return data
            if not d.options:
                return self._fallback(d)

    def _choose(self, d):
        view = self.state.view()
        sig = None
        if d.selection is None or not d.selection.picked:
            sig = (d.raw, self.state.turn, self.state.phase, self._signature(view))
            for i, seen in self._visits.get(sig, ()):
                if seen == self.state.waiting:
                    self._noops.setdefault(sig, set()).add(i)
            for i in self._noops.get(sig, ()):
                if i < len(d.options) and d.options[i].kind not in PROGRESS_KINDS:
                    d.options[i].noop = True
        idx = self.policy(view, d)
        if sig is not None:
            self._visits.setdefault(sig, set()).add((idx, self.state.waiting))
        return idx

    @staticmethod
    def _signature(v):
        cards = tuple(sorted((c["rel"], c["loc"], c["seq"], c["code"], c["pos"], c["atk"], c["def"], c["overlay"],
                              tuple(sorted(c["counters"].items()))) for c in v["cards"]))
        players = tuple(tuple(sorted(p.items())) for p in v["players"])
        chain = tuple((c["code"], c["rel"]) for c in v["chain"])
        return cards, players, chain

    def _fill_codes(self, d):
        """EDOPro blanks the codes of the opponent's cards in card selections; the ones this player
        can see on the field are filled back in, as the simulator shows them."""
        def fill(c):
            if not c or c[0]:
                return c
            code, con, loc, seq = c
            card = self.state._get(con, loc, seq)
            if card and card.get("code") and (loc not in (LOCATION_HAND, LOCATION_DECK) or card.get("public")):
                return (card["code"], con, loc, seq)
            return c
        d.cards = [fill(c) for c in d.cards]
        for o in d.options:
            o.card = fill(o.card)
        if d.selection is not None:
            d.selection.refresh_options()

    def _fallback(self, d):
        if d.msg in (MSG_SELECT_CARD, MSG_SELECT_TRIBUTE) and d.cards:
            from ..core.messages import encode_card_list
            return encode_card_list(list(range(max(1, d.min))))
        if d.options:
            return encode_response(d, d.options[0])
        return encode_response(d, Option(0, value=0))
