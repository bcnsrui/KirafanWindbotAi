"""Tactical look-ahead for the rest of our own turn (until our end phase: the opponent's 3-card draw
makes anything further a guess).

From the view (through model.features._Tactics) it plans our remaining actions this turn:
  - dotte effects of our battle creamates (within the gauge; a dotte turns its user stiff, and a stiff
    creamate has none - except one that its normal attack turns upright again, 혼다 타마키), with what
    they bring besides damage: 나약함 (HP 1 falls), attack for their user, extra normal attacks for a
    further stiff creamate, and their own "추가 경직" cost,
  - normal attacks (each attack counter; the first attack of a creamate adds 1 to the gauge after it;
    "[자신 배틀 페이즈 개시시]" attack bonuses still to come, a "달 속성에게 공격력+1" passive, 절반 all-enemy
    attackers, taunts and 봇치 limiting the targets),
  - attack buffs from our hand put on right before a normal attack (supports cannot be played while
    attacking; a "다음 공격이 1번만" buff is spent by that attack, other buffs last the turn), and gauge /
    extra-attack supports from our hand,
against an opponent who answers from its hidden hand whenever one of our actions would defeat one of its
creamates - with what hurts us most (minimax): nothing yet (keeping its cards for later), a single /
all-ally heal or overheal on any of them, a cleanse chained after a heal for an 불행 one, or 수면 on our
attacker / all of ours (with an all-ally heal: 야마구치 키사라기) - with up to OPP_HEALS_MAX cards, within its
free support slots, sanctuary room and gauge. A creamate of its that falls frees its level of that room,
so an answer can turn up after a first kill (two falling to one all-enemy hit cannot be saved that way).

The plan maximises defeated enemy battle creamates first; then cards drawn (worth more with a small
hand) and damage on survivors (CHIP: their owner often trades a damaged creamate in anyway); and spends
as few cards and as little gauge as it needs. plan(tac) returns the values of the actions the decision's options map to.
"""
import itertools
import re

from ..cardtext import text_of
from ..core.constants import POS_DEFENSE, COUNTER_BIND, COUNTER_SLEEP, COUNTER_WEAK

COUNTER_ALL_ATTACK = 0xc04   # (model.features): the next normal attack hits every enemy

OPP_HEAL = 4          # a plain single heal from the opponent's hidden hand ...
OPP_ALL_HEAL = 2      # ... or an all-ally heal,
OPP_OVER_HEAL = 3     # ... or a single heal past the maximum HP (나츠메 노노, 유노, 히로, 샤미코),
OPP_OVER_ALL = 2      # ... or an all-ally one (유노 / 샤미코 dotte: level 2, gauge 1+)
OPP_HEALS_MAX = 2     # it can answer with at most this many cards in one turn
OPP_HEAL_LEVEL = 1    # level of those heals (sanctuary room they need: 1-level priests, 나츠메 노노)
OPP_OVER_ALL_LEVEL = 2
# its other answers in the card pool: 수면 (attack -1 for the turn) on the attacker (마나카 아오, level 1) or on
# all of ours (니죠 오미, level 2), both with an all-ally heal of 2 (야마구치 키사라기 dotte: level 2, gauge 2);
# and a cleanse (level 1, 타케야 유키 and others) chained after a heal lets it work on an 불행 creamate
OPP_SLEEP_LEVEL = 1
OPP_SLEEP_ALL_LEVEL = 2
OPP_KISARAGI_GAUGE = 2
OPP_CLEANSE_LEVEL = 1
KILL = 1000           # score: an enemy battle creamate defeated
CARD = 15             # ... a buff card spent
GAUGE = 3             # ... a gauge point spent
# ... an HP removed from an enemy that survives the turn: little, as its owner may well send a damaged
# creamate to the dotte zone at its standby phase and call a fresh one (whose entry effect pays off), so
# chip damage counts only through a kill it leads to this turn. (Overkill is worth nothing.)
CHIP = 3
# ... a card a dotte draws: DRAW * (DRAW_HAND - hand size), nothing with a full hand. A 1-damage + draw
# dotte beats a 3-damage one when both defeat the same enemy, and with a small hand also when neither does.
DRAW = 3
DRAW_HAND = 6
MAX_STATES = 1500     # look-ahead budget per board (Defense)
PLAN_STATES = 600     # ... and per look-ahead of Plan (its opponent's answers branch too)

_DRAW = re.compile(r"드로우\s*\+\s*(\d+)")


def dotte_draws(db, code):
    """Cards the battle creamate's dotte effect draws (its 돗테오키 게이지 section of the text)."""
    card = db.get(code)
    if card is None:
        return 0
    for part in card.desc.split("[")[1:]:
        if part.startswith("돗테오키 게이지"):
            return sum(int(n) for n in _DRAW.findall(part))
    return 0


class Plan:
    """Best scores of our remaining turn. Actions, with indices into mine / foes:
      ("dotte", b, e, paid, more, opt): b's dotte on enemy e (None: every enemy); paid: the creamates turned
          stiff for its "추가 경직" (its own, its optional cost's, the one for its extra attacks), more: who gets
          those extra normal attacks (an index into mine, "all", or None: not taken), opt: its optional cost
          for extra damage paid
      ("attack", a, e, buff, paid): a's normal attack on e with hand buff `buff` (index into buffs or None)
          right before it, paid: the creamates paying that buff's stiff cost
      ("tool", k, a, paid): hand support k (index into tools): gauge, or extra normal attacks for a (None:
          every ally)"""

    def __init__(self, tac):
        self.tac = tac
        self.mine = list(tac.battle[0].values())
        self.foes = list(tac.battle[1].values())
        self.taunt = tuple(tac.is_taunt(f) for f in self.foes)
        self.untouchable = tuple(text_of(tac.db, f["code"]).not_attackable for f in self.foes)   # 봇치
        texts = [text_of(tac.db, a["code"]) for a in self.mine]
        powers = [tac.attack_of(a) for a in self.mine]
        self.atk0 = tuple(p[0] for p in powers)          # with the battle-start bonuses still to come
        self.once0 = tuple(p[1] for p in powers)         # ... those on its next attack only
        self.half = tuple(t.half_all for t in texts)     # 절반 all-enemy attackers
        self.unstiff = tuple(t.unstiff_after_attack for t in texts)
        self.attack_gauge = tuple(t.attack_gauge for t in texts)   # its first attack's gauge (소냐 3)
        self.attack_weak = tuple(t.attack_weak for t in texts)     # its attacks give 나약함 (시마 린)
        self.vs = tuple(tuple(tac.hit_of(a, f, 0) for f in self.foes) for a in self.mine)   # 달 속성 passive
        # per battle creamate of ours: (gauge cost, hits every enemy, damage per enemy, draws, 나약함 ("one" /
        # "all"), (attack it gives its user, + per standing enemy?), extra attacks (stiff, whose, +N), its
        # extra stiff, enemy count in the damage, optional cost (extra gauge, extra stiff, damage per enemy
        # with it) or None) or None
        self.dottes = []
        for b, t in zip(self.mine, texts):
            if t.dotte_hits:
                var = len(self.foes) if t.dotte_var and t.dotte_var[1] == "opp_battle" else 0
                dmg = [tac.dotte_damage(t, b, f) for f in self.foes]
                opt = None
                if t.dotte_options:
                    extra_g = sum(int(v[0]) for name, v in t.dotte_options if name == "opt_gauge")
                    extra_s = sum(int(v[0]) for name, v in t.dotte_options if name == "opt_stiff")
                    if any(name == "opt_stiff_each" for name, _ in t.dotte_options):
                        extra_s = sum(1 for c in self.mine if c is not b and not c["pos"] & POS_DEFENSE)
                    dmg_opt = [tac.dotte_damage(t, b, f, True) for f in self.foes]
                    opt = (extra_g, extra_s, dmg_opt) if dmg_opt != dmg else None
                self.dottes.append((t.dotte, t.dotte_to_all, dmg, dotte_draws(tac.db, b["code"]), t.dotte_weak,
                                    (t.dotte_self_atk, t.dotte_self_atk_opp), t.dotte_more_attacks, t.dotte_stiff, var,
                                    opt))
            else:
                self.dottes.append(None)
        self.buffs = tac.hand_buffs()   # [(code, amount, all allies, one attack only, level, gauge, stiff cost)]
        self.tools = tac.hand_tools()   # [(code, "gauge" / "attack", +N, whose, level, gauge, stiff cost)]
        self.bound = tuple(b["counters"].get(COUNTER_BIND, 0) > 0 for b in self.mine)   # 속박: no dotte
        self.draw_value = DRAW * max(0, DRAW_HAND - tac.view["players"][0].get("hand", 0))
        self.base_hp = tuple(f["def"] for f in self.foes)
        self.weak0 = tuple(f["counters"].get(COUNTER_WEAK, 0) > 0 for f in self.foes)   # 나약함: HP 1 is lethal
        self.max_hp = tuple(f["base_def"] for f in self.foes)
        self.no_heal = tuple(f["counters"].get(0xb02, 0) > 0 for f in self.foes)        # 불행
        self.levels = tuple(f["level"] for f in self.foes)
        # the all-ally overheal (유노 / 샤미코 dotte) costs gauge and one of its battle creamates turned stiff;
        # its creamates stiffened on its own turn stay so through ours
        p1 = tac.der["players"][1]
        self.opp_over_all = p1["dotte"] >= 1 and any(not f["pos"] & POS_DEFENSE for f in self.foes)
        self.opp_room = 99 if tac.view["players"][1].get("summoner_code") == 10050110 else p1["margin"]
        self.opp_slots = 5 - tac.slots_used[1]
        self.memo = {}
        self.values = {}   # (state, first action) -> its outcome (best / value)
        self.root = None
        self.states = 0

    def start(self, opp_cards):
        """Start state; opp_cards: cards the opponent is taken to hold to answer with (0: it cannot answer).
        (hp, attacks, first attack still to come, dotte used, attack on the next attack only, gauge, hand
        buffs left, free support slots, sanctuary room, opponent (cards, slots, room, gauge), can still be
        turned stiff, attack for the rest of the turn, 나약함, hand tools left, next attack hits all, 수면)"""
        attacks = tuple(self.tac._attacks_left(a) for a in self.mine)
        first = tuple(n > 0 for n in attacks)
        p0 = self.tac.der["players"][0]
        room = 99 if self.tac.view["players"][0].get("summoner_code") == 10050110 else p0["margin"]
        slots = 5 - self.tac.slots_used[0]
        stand = tuple(not a["pos"] & POS_DEFENSE for a in self.mine)
        all_next = tuple(a["counters"].get(COUNTER_ALL_ATTACK, 0) > 0 for a in self.mine)
        asleep = tuple(a["counters"].get(COUNTER_SLEEP, 0) > 0 for a in self.mine)
        n = len(self.mine)
        # our dottes still waiting on the chain land first (the opponent answers them as it likes best)
        hp, opp = self.base_hp, (opp_cards, self.opp_slots, self.opp_room, self.tac.der["players"][1]["dotte"])
        waiting = {e: d for e, d in enumerate(self.tac.chain_damage(f) for f in self.foes) if d}
        if waiting:
            landed = [self._land(h, o, waiting, self.weak0) for h, o, _ in
                      self._answers(hp, opp, lambda cut: waiting, self.weak0, None, asleep)]
            hp, opp, _ = min(landed, key=lambda r: (sum(1 for x in r[0] if x <= 0), -sum(r[0])))
        self.pending_kills = sum(1 for e in range(len(hp)) if hp[e] <= 0)
        return (hp, attacks, first, (False,) * n, self.once0, p0["dotte"] + self.tac.pending_attack_gauge(0),
                (True,) * len(self.buffs), slots, room, opp, stand, (0,) * n, self.weak0, (True,) * len(self.tools),
                all_next, asleep)

    # ---- one action --------------------------------------------------
    def _falls(self, hp, e, dmg, weak):
        return hp[e] > 0 and hp[e] - dmg <= (1 if weak[e] else 0)   # (나약함: HP 1 is lethal)

    def _healed(self, hp, e, n, over=False, cleanse=False):
        if hp[e] <= 0 or (self.no_heal[e] and not cleanse):
            return hp[e]
        return hp[e] + n if over else max(hp[e], min(hp[e] + n, self.max_hp[e]))

    def _answers(self, hp, opp, hits_for, weak, attacker, asleep):
        """[(hp after the opponent's heals, opp after paying for them, 수면 put on: None / "one" / "all")]: what
        it may answer one action with before it resolves. hits_for(cut): the action's {foe: damage} with its
        attacker's attack cut by cut; attacker: the normal attacker (index into mine), None for a dotte (수면
        does not touch it). Nothing; or, while someone would still fall and it has cards (OPP_HEALS_MAX at
        most), slots, room and gauge for it: a single heal / overheal on a falling one, an all-ally heal /
        overheal, either with a cleanse chained after it for an 불행 one (one card more), 수면 on the
        attacker or on all of ours, or 키사라기 (all-ally heal + 수면 on all)."""
        out = [(hp, opp, None)]
        if not opp[0] or not opp[1] or not any(self._falls(hp, e, d, weak) for e, d in hits_for(0).items()):
            return out   # (nobody falls, or it has nothing to answer with)
        frontier = out
        for _ in range(OPP_HEALS_MAX):
            nxt = []
            for h, o, slept in frontier:
                cut = 1 if slept and attacker is not None and not asleep[attacker] else 0
                falling = [e for e, d in hits_for(cut).items() if self._falls(h, e, d, weak)]
                if falling:
                    nxt += self._answer(h, o, falling, attacker, asleep, slept)
            out = out + nxt
            frontier = nxt
        return self._undominated(list(dict.fromkeys(out)))

    @staticmethod
    def _undominated(answers):
        """The answers no other one beats for the opponent: at least as much HP on every creamate, as many
        cards, slots and room left, and 수면 at least as wide (its gauge left is not weighed)."""
        rank = {None: 0, "one": 1, "all": 2}
        keep = []
        for i, (h, o, sl) in enumerate(answers):
            beaten = False
            for j, (h2, o2, sl2) in enumerate(answers):
                if j == i or rank[sl2] < rank[sl] or any(o2[k] < o[k] for k in range(3)) \
                        or any(x2 < x for x2, x in zip(h2, h)):
                    continue
                if (h2, o2[:3], sl2) != (h, o[:3], sl) or j < i:
                    beaten = True
                    break
            if not beaten:
                keep.append((h, o, sl))
        return keep

    def _answer(self, h, o, falling, attacker, asleep, slept):
        cards, slots, room, gauge = o

        def pay(n_cards, level, g=0):
            if cards < n_cards or slots < n_cards or room < level or gauge < g:
                return None
            return cards - n_cards, slots - n_cards, room - level, gauge - g

        def one(e, n, over, cleanse):
            return tuple(self._healed(h, f, n, over, cleanse) if f == e else h[f] for f in range(len(h)))

        def every(n, over, cleanse):
            return tuple(self._healed(h, f, n, over, cleanse) for f in range(len(h)))

        out = []
        unlucky = any(self.no_heal[e] for e in falling)
        for cleanse in ((False, True) if unlucky else (False,)):
            extra = (1, OPP_CLEANSE_LEVEL) if cleanse else (0, 0)
            o2 = pay(1 + extra[0], OPP_HEAL_LEVEL + extra[1])
            if o2:
                for e in falling:
                    if self.no_heal[e] and not cleanse:
                        continue
                    out += [(one(e, OPP_HEAL, False, cleanse), o2, slept), (one(e, OPP_OVER_HEAL, True, cleanse), o2, slept)]
                out.append((every(OPP_ALL_HEAL, False, cleanse), o2, slept))
            o2 = pay(1 + extra[0], OPP_OVER_ALL_LEVEL + extra[1], 1)
            if o2 and self.opp_over_all:
                out.append((every(OPP_OVER_ALL, True, cleanse), o2, slept))
        if not slept:
            o2 = pay(1, OPP_SLEEP_LEVEL)
            if o2 and attacker is not None and not asleep[attacker]:
                out.append((h, o2, "one"))
            o2 = pay(1, OPP_SLEEP_ALL_LEVEL)
            if o2:
                out.append((h, o2, "all"))
            o2 = pay(1, OPP_SLEEP_ALL_LEVEL, OPP_KISARAGI_GAUGE)
            if o2:
                out.append((every(OPP_ALL_HEAL, False, False), o2, "all"))
        return out

    def _land(self, hp, opp, hits, weak):
        """(hp, opp, damage taken by survivors) once hits {foe: dmg} land; every creamate that falls frees its
        level of the opponent's sanctuary room for what comes after."""
        hp = list(hp)
        cards, slots, room, gauge = opp
        chip = 0
        for e, d in hits.items():
            if hp[e] <= 0:
                continue
            if self._falls(hp, e, d, weak):
                hp[e] = 0
                room += self.levels[e]   # (its level leaves the opponent's field)
            else:
                chip += d
                hp[e] -= d
        return tuple(hp), (cards, slots, room, gauge), chip

    def _stiff_choices(self, cands, n):
        """Sets of n of the creamates cands to turn stiff; those that lose nothing by it (no dotte, 속박) count
        as one choice. [] when there are not enough."""
        if n <= 0:
            return [()]
        if len(cands) < n:
            return []
        free = [i for i in cands if self.dottes[i] is None or self.bound[i]]
        rest = [i for i in cands if i not in free]
        if len(free) >= n:
            return [tuple(free[:n])]
        return [tuple(free) + c for c in itertools.combinations(rest, n - len(free))]

    def _key(self, e, hp, weak):
        return (hp[e], self.max_hp[e], self.levels[e], self.no_heal[e], weak[e], self.taunt[e], self.untouchable[e],
                tuple(v[e] for v in self.vs), tuple(d[2][e] if d else 0 for d in self.dottes),
                tuple(d[9][2][e] if d and d[9] else 0 for d in self.dottes))

    def _distinct(self, foes, hp, weak):
        """foes without the ones that are the same as an earlier one to every action (same HP, maximum HP,
        level, 불행, 나약함, taunt, 봇치, attribute bonus and dotte damage of every attacker): one branch for
        them all."""
        seen, out = set(), []
        for e in foes:
            key = self._key(e, hp, weak)
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    def same_as(self, s, e):
        """The enemy whose branches stand for enemy e in state s (actions name only one of identical ones)."""
        hp, weak = s[0], s[12]
        key = self._key(e, hp, weak)
        return next((f for f in range(len(self.foes)) if hp[f] > 0 and self._key(f, hp, weak) == key), e)

    def actions(self, s):
        hp, attacks, first, dotted, pend, gauge, hand, slots, room, opp, stand, up, weak, tools, all_next, asleep = s
        out = []
        alive = [e for e in range(len(self.foes)) if hp[e] > 0]
        if not alive:
            return out
        picks = self._distinct(alive, hp, weak)   # targets worth a branch of their own
        standing = tuple(i for i in range(len(stand)) if stand[i])
        for b, d in enumerate(self.dottes):
            # (Kirafan6.damcon: no dotte from a stiff or 속박 creamate - a stiff cost paid earlier blocks it)
            if d is None or dotted[b] or d[0] > gauge or not stand[b] or self.bound[b]:
                continue
            others = tuple(i for i in standing if i != b)
            more, opt = d[6], d[9]
            for use_opt in ((False, True) if opt else (False,)):
                if use_opt and d[0] + opt[0] > gauge:
                    continue
                for e in ((None,) if d[1] else picks):
                    for paid in self._stiff_choices(others, d[7] + (opt[1] if use_opt else 0)):
                        out.append(("dotte", b, e, paid, None, use_opt))
                        if not more:
                            continue
                        left = tuple(i for i in others if i not in paid)
                        whose = (b,) if more[1] == "self" else (range(len(self.mine)) if more[1] == "one" else ("all",))
                        for extra in self._stiff_choices(left, more[0]):
                            out += [("dotte", b, e, paid + extra, r, use_opt) for r in whose]
        taunts = [e for e in alive if self.taunt[e]]
        targets = self._distinct(taunts, hp, weak) if taunts else picks
        if len(alive) > 1:
            targets = [e for e in targets if not self.untouchable[e]] or targets
        for a in range(len(self.mine)):
            if not attacks[a]:
                continue
            # (a 절반 / all-enemy attack hits everyone whichever target it declares)
            for e in (targets[:1] if self.half[a] or all_next[a] else targets):
                out.append(("attack", a, e, None, ()))
                for k, (code, n, to_all, once, level, cost, stiff) in enumerate(self.buffs):
                    if not (hand[k] and slots > 0 and level <= room and cost <= gauge):
                        continue
                    # which creamates pay a "추가 경직" cost: every choice is tried, a stiff one loses its dotte
                    # (the search weighs that with the gauge the attacks will still bring)
                    out += [("attack", a, e, k, paid) for paid in self._stiff_choices(standing, stiff)]
        for k, (code, kind, n, who, level, cost, stiff) in enumerate(self.tools):
            if not (tools[k] and slots > 0 and level <= room and cost <= gauge):
                continue
            if kind == "gauge" and not any(d and not dotted[b] and stand[b] and not self.bound[b]
                                           and gauge < d[0] + (d[9][0] if d[9] else 0) <= gauge - cost + n
                                           for b, d in enumerate(self.dottes)):
                continue   # (more gauge only matters for a dotte it cannot pay yet)
            for paid in self._stiff_choices(standing, stiff):
                out += [("tool", k, r, paid) for r in ((None,) if kind == "gauge" or who == "all"
                                                        else range(len(self.mine)))]
        return out

    def step(self, s, act):
        """[(next state, immediate score)]: one per answer the opponent may give to our action act."""
        hp, attacks, first, dotted, pend, gauge, hand, slots, room, opp, stand, up, weak, tools, all_next, asleep = s
        attacks, first, dotted, pend, stand, up, all_next = (list(x) for x in (attacks, first, dotted, pend, stand, up,
                                                                               all_next))
        score = 0
        alive = [f for f in range(len(self.foes)) if hp[f] > 0]
        attacker = None
        if act[0] == "dotte":
            _, b, e, paid, more_to, use_opt = act
            cost, to_all, dmg, draws, weak_kind, self_up, more, _, var, opt = self.dottes[b]
            if use_opt:
                cost, dmg = cost + opt[0], opt[2]
            gauge -= cost
            score += draws * self.draw_value - cost * GAUGE
            targets = alive if e is None else [e]
            up[b] += self_up[0] + (len(alive) if self_up[1] else 0)
            if weak_kind:
                hit_weak = alive if weak_kind == "all" else targets
                weak = tuple(w or f in hit_weak for f, w in enumerate(weak))
            fixed = {f: max(0, dmg[f] - var + len(alive)) for f in targets}
            hits_for = lambda cut: fixed  # noqa: E731
            dotted[b] = True
            stand[b] = False   # its dotte cost turns it stiff
            for i in paid:
                stand[i] = False
            if more_to is not None:
                for i in (range(len(self.mine)) if more_to == "all" else (more_to,)):
                    attacks[i] += more[2]
        elif act[0] == "attack":
            _, a, e, k, paid = act
            attacker = a
            if k is not None:
                code, n, to_all, once, level, cost, stiff = self.buffs[k]
                hand = hand[:k] + (False,) + hand[k + 1:]
                slots -= 1
                room -= level
                gauge -= cost
                for i in paid:
                    stand[i] = False
                score -= CARD + cost * GAUGE
                for i in (range(len(self.mine)) if to_all else (a,)):
                    if once:
                        pend[i] += n
                    else:
                        up[i] += n
            atk = self.atk0[a] + up[a] + pend[a]
            if self.half[a]:
                hits_for = lambda cut: {f: max(0, atk - cut) // 2 for f in alive}  # noqa: E731
            elif all_next[a]:
                hits_for = lambda cut: {f: max(0, atk - cut + self.vs[a][f]) for f in alive}  # noqa: E731
                all_next[a] = False
            else:
                hits_for = lambda cut: {e: max(0, atk - cut + self.vs[a][e])}  # noqa: E731
            if self.attack_weak[a]:   # (its 나약함 lands with the damage: HP 1 falls at once)
                weak = tuple(w or f in hits_for(0) for f, w in enumerate(weak))
            pend[a] = 0
            attacks[a] -= 1
            if first[a]:
                gauge += self.attack_gauge[a]
                first[a] = False
            if self.unstiff[a] and not stand[a]:   # (혼다 타마키: upright again, its dotte usable again)
                stand[a] = True
                dotted[a] = False
        else:
            _, k, r, paid = act
            code, kind, n, who, level, cost, stiff = self.tools[k]
            tools = tools[:k] + (False,) + tools[k + 1:]
            slots -= 1
            room -= level
            gauge -= cost
            for i in paid:
                stand[i] = False
            score -= CARD + cost * GAUGE
            if kind == "gauge":
                gauge += n
            else:
                for i in (range(len(self.mine)) if r is None else (r,)):
                    attacks[i] += n
            return [((hp, tuple(attacks), tuple(first), tuple(dotted), tuple(pend), gauge, hand, slots, room,
                      opp, tuple(stand), tuple(up), weak, tools, tuple(all_next), asleep), score)]
        rest = (tuple(attacks), tuple(first), tuple(dotted), tuple(pend), gauge, hand, slots, room)
        before = sum(1 for h in hp if h <= 0)
        out = []
        for h, o, slept in self._answers(hp, opp, hits_for, weak, attacker, asleep):
            cut = 1 if slept and attacker is not None and not asleep[attacker] else 0
            h2, o2, removed = self._land(h, o, hits_for(cut), weak)
            up2, asleep2 = up, asleep
            if slept:   # (수면: attack -1 for the turn, once)
                who = range(len(self.mine)) if slept == "all" else (attacker,)
                up2 = tuple(x - (1 if i in who and not asleep[i] else 0) for i, x in enumerate(up))
                asleep2 = tuple(z or i in who for i, z in enumerate(asleep))
            kills = [i for i in range(len(h2)) if h2[i] <= 0 and hp[i] > 0]
            sc = score + (sum(1 for x in h2 if x <= 0) - before) * KILL + sum(100 * self.levels[i] for i in kills) \
                + CHIP * removed
            out.append(((h2,) + rest + (o2, tuple(stand), tuple(up2), weak, tools, tuple(all_next), asleep2), sc))
        return out

    def _outcome(self, s, act):
        """(score, kills) of act and the best play after it, against the opponent's worst answer for us."""
        res = None
        for nxt, sc in self.step(s, act):
            fut = self._best(nxt)
            k = sum(1 for i in range(len(nxt[0])) if nxt[0][i] <= 0 and s[0][i] > 0)
            cand = (sc + fut[0], k + fut[1])
            if res is None or cand[0] < res[0]:
                res = cand
        return res

    def best(self, s):
        """(best score from s on, its kills); a look-ahead of its own (PLAN_STATES new boards). The outcome of
        each first action is kept for value()."""
        self.states = 0
        self.root = s
        return self._best(s)

    def _best(self, s):
        got = self.memo.get(s)
        if got is not None:
            return got
        self.states += 1
        if self.states >= PLAN_STATES:
            return 0, 0   # (out of budget: not remembered, a later look-ahead may finish it)
        res = (0, 0)
        root = s is self.root
        for act in self.actions(s):
            cand = self._outcome(s, act)
            if root:
                self.values[(s, act)] = cand
            if cand[0] > res[0]:
                res = cand
        self.memo[s] = res
        return res

    def value(self, s, act):
        got = self.values.get((s, act))
        if got is None:
            self.states = 0
            got = self.values[(s, act)] = self._outcome(s, act)
        return got

    def first_step(self, s):
        best, out = None, None
        for act in self.actions(s):
            v = self.value(s, act)
            if best is None or v[0] > best[0]:
                best, out = v, act
        return out, best or (0, 0)


# ---- the opponent's turn: what to answer with ---------------------------
LOSE = 1000           # one of our battle creamates defeated (+100 per level)
DEF_CARD = 20         # a card of ours spent, + DEF_CARD_SHORT per card below DEF_HAND in our hand
DEF_CARD_SHORT = 15
DEF_HAND = 5
DEF_HP = 2            # an HP lost by a creamate of ours that survives (only breaks ties)


class Defense:
    """The rest of the opponent's turn (until its end phase). It plays to defeat our battle creamates:
    its normal attacks left (taunt and 봇치 respected, all-enemy and 절반 all-enemy attacks, "[자신 배틀 페이즈
    개시시]" bonuses still to come, a "달 속성에게 공격력+1" passive, the biggest hidden attack buff it could
    still play once before an attack when hidden_buff), and its dottes (in attack position, no 속박, within
    the gauge its first attacks still add to; with 나약함, the attack they give, the extra attacks they
    bring for a further stiff creamate, their own extra stiff; a dotte turns its user stiff, 혼다 타마키's
    normal attack turns it upright again). We answer each of its actions with at most one of our hand
    cards (or a heal + cleanse pair for an 불행 ally, chained so the cleanse resolves first): a single /
    all-ally heal (overheal too), or 수면 on the attacker / all of them; within our free support slots,
    sanctuary room (a fallen creamate of ours frees its level) and gauge (dotte effects of supports).
    Minimax with memo: (our loss score, our creamates defeated)."""

    def __init__(self, tac, hidden_buff=False, resolving=None):
        """resolving: the opponent's battle creamate whose dotte is on the chain (already stiff from its
        cost, but its effect still to resolve)."""
        self.tac = tac
        self.mine = list(tac.battle[0].values())
        self.foes = list(tac.battle[1].values())
        self.taunt = [i for i, c in enumerate(self.mine) if tac.is_taunt(c)]
        self.untouchable = tuple(text_of(tac.db, c["code"]).not_attackable for c in self.mine)   # 봇치
        self.max_hp = tuple(c["base_def"] for c in self.mine)
        self.levels = tuple(c["level"] for c in self.mine)
        texts = [text_of(tac.db, f["code"]) for f in self.foes]
        powers = [tac.attack_of(f) for f in self.foes]
        self.atk = tuple(p[0] for p in powers)
        self.once = tuple(p[1] for p in powers)    # on its first attack still to come
        self.half = tuple(t.half_all for t in texts)
        self.unstiff = tuple(t.unstiff_after_attack for t in texts)
        self.attack_gauge = tuple(t.attack_gauge for t in texts)
        self.attack_weak = tuple(t.attack_weak for t in texts)
        self.vs = tuple(tuple(tac.hit_of(f, c, 0) for c in self.mine) for f in self.foes)
        self.all_attack = tuple(f["counters"].get(COUNTER_ALL_ATTACK, 0) > 0 for f in self.foes)
        # per opponent battle creamate: (gauge cost, hits every one of ours, damage per creamate of ours,
        # 나약함, (attack for its user, + per creamate of ours?), extra attacks (stiff, whose, +N), its extra
        # stiff, count of ours in the damage) or None
        self.dottes = []
        for f, t in zip(self.foes, texts):
            if t.dotte_hits and (tac._can_dotte(f) or f is resolving):
                var = len(self.mine) if t.dotte_var and t.dotte_var[1] == "opp_battle" else 0
                # (its optional extra-damage cost taken as paid)
                self.dottes.append((t.dotte, t.dotte_to_all, [tac.dotte_damage(t, f, c, True) for c in self.mine], t.dotte_weak,
                                    (t.dotte_self_atk, t.dotte_self_atk_opp), t.dotte_more_attacks, t.dotte_stiff, var))
            else:
                self.dottes.append(None)
        self.buff = max(tac.opp_buff) if hidden_buff else 0
        # [(hand card index, kind, amount, all, overheal, level, gauge, 수면 with it: None / "one" / "all", code)]
        self.cards = tac.defense_cards()
        self.n_hand = 1 + max((c[0] for c in self.cards), default=-1)
        self.cleanses = len(tac.hand_cleanses)
        hand = tac.view["players"][0].get("hand", 0)
        self.card_cost = DEF_CARD + DEF_CARD_SHORT * max(0, DEF_HAND - hand)
        self.memo = {}
        self.states = 0

    def start(self):
        """(our hp, 불행, room, slots, gauge, hand cards left, cleanses left, its attacks, first attack still to
        come, dotte used, its gauge, hidden buff left, 수면 cuts, attack its dottes gave, its creamates that can
        still be turned stiff, 나약함 on ours)"""
        p0 = self.tac.der["players"][0]
        room = 99 if self.tac.view["players"][0].get("summoner_code") == 10050110 else p0["margin"]
        attacks = tuple(self.tac._attacks_left(f) for f in self.foes)
        return (tuple(c["def"] for c in self.mine), tuple(c["counters"].get(0xb02, 0) > 0 for c in self.mine),
                room, 5 - self.tac.slots_used[0], p0["dotte"], (True,) * self.n_hand, self.cleanses,
                attacks, tuple(n > 0 for n in attacks), (False,) * len(self.foes),
                self.tac.der["players"][1]["dotte"], self.buff > 0, (0,) * len(self.foes), (0,) * len(self.foes),
                tuple(not f["pos"] & POS_DEFENSE for f in self.foes),
                tuple(c["counters"].get(COUNTER_WEAK, 0) > 0 for c in self.mine))

    # ---- the opponent's actions ----------------------------------------
    def moves(self, s):
        hp, unhappy, room, slots, gauge, hand, cleanses, attacks, first, dotted, ogauge, buff, cut, up, fstand, weak = s
        alive = [i for i in range(len(hp)) if hp[i] > 0]
        if not alive:
            return []
        targets = [i for i in self.taunt if hp[i] > 0] or alive
        if len(alive) > 1:
            targets = [i for i in targets if not self.untouchable[i]] or targets
        out = []
        for b in range(len(self.foes)):
            if attacks[b]:
                for t in (None,) if self.all_attack[b] or self.half[b] else targets:
                    out.append(("attack", b, t, False))
                    if buff:
                        out.append(("attack", b, t, True))
            d = self.dottes[b]
            if d and not dotted[b] and d[0] <= ogauge and fstand[b]:
                others = sum(1 for j in range(len(self.foes)) if fstand[j] and j != b)
                if d[6] > others:
                    continue
                more = d[5]
                whose = () if not more or more[0] + d[6] > others else \
                    ((b,) if more[1] == "self" else (range(len(self.foes)) if more[1] == "one" else ("all",)))
                for t in ((None,) if d[1] else alive):
                    out.append(("dotte", b, t, None))
                    out += [("dotte", b, t, r) for r in whose]
        return out

    def hits(self, s, move, cut):
        """{our creamate: damage} of the opponent's move (after 수면 cuts)."""
        hp, first, up = s[0], s[8], s[13]
        kind, b, t = move[0], move[1], move[2]
        alive = [i for i in range(len(hp)) if hp[i] > 0]
        if kind == "attack":
            atk = self.atk[b] + up[b] + (self.once[b] if first[b] else 0) + (self.buff if move[3] else 0) - cut[b]
            if self.half[b]:
                return {i: max(0, atk) // 2 for i in alive}
            return {i: max(0, atk + self.vs[b][i]) for i in (alive if t is None else [t])}
        d = self.dottes[b]
        return {i: max(0, d[2][i] - d[7] + len(alive)) for i in (alive if t is None else [t])}

    # ---- our answers ---------------------------------------------------
    def answers(self, s, move):
        """[(card indices used, heal {creamate: HP}, heal on 불행 allowed, 수면 on: None / "one" / "all")]"""
        hp, unhappy, room, slots, gauge, hand, cleanses = s[:7]
        out = [((), {}, False, None)]
        hit = list(self.hits(s, move, s[12]))
        alive = {i: None for i in range(len(hp)) if hp[i] > 0}
        for k, (h, kind, n, to_all, over, level, cost, sleep, _code) in enumerate(self.cards):
            if not hand[h] or slots < 1 or level > room or cost > gauge:
                continue
            sl = sleep if move[0] == "attack" else None
            if kind == "heal":
                for heal in ([{i: n for i in alive}] if to_all else [{i: n} for i in hit]):
                    out.append(((k,), heal, False, sl))
                    if any(unhappy[i] for i in heal) and cleanses and slots >= 2 and room >= level + 1:
                        out.append(((k, "cleanse"), heal, True, sl))
            elif kind == "sleep" and move[0] == "attack":
                out.append(((k,), {}, False, "all" if to_all else "one"))
        return out

    def step(self, s, move, ans):
        """(next state, loss score of this exchange)."""
        hp, unhappy, room, slots, gauge, hand, cleanses, attacks, first, dotted, ogauge, buff, cut, up, fstand, weak = s
        used, heal, through_unhappy, sleep = ans
        loss = 0
        over_k = False
        hp, unhappy, hand, cut = list(hp), list(unhappy), list(hand), list(cut)
        for k in used:
            if k == "cleanse":
                cleanses -= 1
                slots -= 1
                room -= 1
                loss += self.card_cost
                continue
            h, kind, n, to_all, over, level, cost, _, _code = self.cards[k]
            hand[h] = False
            slots -= 1
            room -= level
            gauge -= cost
            loss += self.card_cost
            over_k = over
        for i, n in heal.items():
            if unhappy[i] and not through_unhappy:
                continue
            if through_unhappy:
                unhappy[i] = False
            new = hp[i] + n if over_k else max(hp[i], min(hp[i] + n, self.max_hp[i]))
            loss -= DEF_HP * (new - hp[i])   # (a heal that restores nothing earns nothing)
            hp[i] = new
        if sleep:
            b = move[1]
            for j in (range(len(self.foes)) if sleep == "all" else (b,)):
                if self.foes[j]["counters"].get(COUNTER_SLEEP, 0) == 0:
                    cut[j] = 1
        b = move[1]
        hits = self.hits((tuple(hp),) + s[1:], move, cut)
        if move[0] == "dotte" and self.dottes[b][3]:
            weak = tuple(w or (i in hits if self.dottes[b][3] == "one" else hp[i] > 0) for i, w in enumerate(weak))
        elif move[0] == "attack" and self.attack_weak[b]:
            weak = tuple(w or i in hits for i, w in enumerate(weak))
        for i, d in hits.items():
            if hp[i] <= 0:
                continue
            if hp[i] - d <= (1 if weak[i] else 0):   # (나약함: HP 1 is lethal)
                hp[i] = 0
                room += self.levels[i]
                loss += LOSE + 100 * self.levels[i]
            else:
                hp[i] -= d
                loss += DEF_HP * d
        attacks, first, dotted, up, fstand = list(attacks), list(first), list(dotted), list(up), list(fstand)
        if move[0] == "attack":
            attacks[b] -= 1
            if move[3]:
                buff = False
            if first[b]:
                first[b] = False
                ogauge += self.attack_gauge[b]
            if self.unstiff[b] and not fstand[b]:   # (혼다 타마키: upright again, its dotte usable again)
                fstand[b] = True
                dotted[b] = False
        else:
            d = self.dottes[b]
            dotted[b] = True
            ogauge -= d[0]
            fstand[b] = False
            up[b] += d[4][0] + (sum(1 for x in s[0] if x > 0) if d[4][1] else 0)
            more_to = move[3] if len(move) > 3 else None
            need = d[6] + (d[5][0] if more_to is not None else 0)
            # the ones it stiffens: those that lose least (no dotte left) first
            for j in sorted((j for j in range(len(self.foes)) if fstand[j] and j != b),
                            key=lambda j: self.dottes[j] is not None and not dotted[j])[:need]:
                fstand[j] = False
            if more_to is not None:
                for j in (range(len(self.foes)) if more_to == "all" else (more_to,)):
                    attacks[j] += d[5][2]
        return (tuple(hp), tuple(unhappy), room, slots, gauge, tuple(hand), cleanses, tuple(attacks), tuple(first),
                tuple(dotted), ogauge, buff, tuple(cut), tuple(up), tuple(fstand), weak), loss

    def exchange(self, s, move, ans):
        """(loss, our creamates lost) of answering move with ans and playing on; a look-ahead of its own
        (MAX_STATES new boards)."""
        self.states = 0
        return self._exchange(s, move, ans)

    def _exchange(self, s, move, ans):
        nxt, loss = self.step(s, move, ans)
        lost = sum(1 for i in range(len(nxt[0])) if nxt[0][i] <= 0 and s[0][i] > 0)
        fut = self._worst(nxt)
        return loss + fut[0], lost + fut[1]

    def worst(self, s):
        """(loss, lost) when the opponent plays its best from s and we answer our best (it may stop)."""
        self.states = 0
        return self._worst(s)

    def _worst(self, s):
        got = self.memo.get(s)
        if got is not None:
            return got
        self.states += 1
        if self.states >= MAX_STATES:
            return 0, 0   # (out of budget: not remembered, a later look-ahead may finish it)
        res = (0, 0)
        for move in self.moves(s):
            v = min((self._exchange(s, move, a) for a in self.answers(s, move)), key=lambda x: x[0])
            if v[0] > res[0]:
                res = v
        self.memo[s] = res
        return res
