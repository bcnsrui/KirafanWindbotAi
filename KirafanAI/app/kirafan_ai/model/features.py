"""Encodes (view, decision) into plain Python lists for the policy network.

Pure Python on purpose: it runs inside the 32-bit simulator workers and inside the 64-bit
inference server for WindBot, producing identical inputs in both.
"""
import collections
import re
from array import array

from ..cards import CLASS_RACES, CLASS_NAMES, ATTRIBUTES, ATTRIBUTE_NAMES
from ..cardtext import text_of, STATUS_COUNTERS
from ..core.constants import *  # noqa: F401,F403
from ..core.messages import (POLICY_MESSAGES, NUM_KINDS, ACTIVATE_KINDS, K_BATTLE_ATTACK, K_CARD_PICK, NOOP_DESCS,
                             K_IDLE_TO_BP, K_IDLE_TO_EP, K_BATTLE_TO_EP, K_BATTLE_TO_M2, K_CHAIN_PASS,
                             K_CARD_CANCEL)
from . import derived

MSG_TYPES = sorted(POLICY_MESSAGES)
CARD_LOCS = [(LOCATION_HAND, False), (LOCATION_MZONE, False), (LOCATION_MZONE, True), (LOCATION_SZONE, False),
             (LOCATION_GRAVE, False), (LOCATION_EXTRA, False), (LOCATION_REMOVED, False), (LOCATION_DECK, False)]
TOKEN_LOCS = (LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE, LOCATION_EXTRA)
MAX_CARD_TOKENS = 64

PLAYER_FEATS = 27
# appended last (model.net.fit_state_dict widens older weights with zeros): support slots used by each
# player, our battle creamates the opponent can still defeat this turn, the worst HP shortfall against that,
# each player can still play a heal from the hand (_Tactics._can_heal), the biggest single / all-ally attack
# buff the opponent could still play from its hidden hand (_Tactics._opp_buff_potential)
GLOBAL_EXTRA_DIM = 8
GLOBAL_DIM = len(PHASES) + 1 + 2 + len(MSG_TYPES) + 3 + 2 * PLAYER_FEATS + 5 + GLOBAL_EXTRA_DIM
G_MY_HAND = len(PHASES) + 1 + 2 + len(MSG_TYPES) + 3 + 2   # our hand size / 10 (_player_feats[2])
G_OPP_HAND = G_MY_HAND + PLAYER_FEATS
G_OPP_CAN_HEAL = GLOBAL_DIM - 3                              # GLOBAL_EXTRA: the opponent can still heal
CARD_DIM = 2 + len(CARD_LOCS) + 1 + 2 + 5 + (len(CLASS_RACES) + 1) + (len(ATTRIBUTES) + 1) + 2 + len(TRACKED_COUNTERS) + 4
OPTION_BASE_DIM = 2 + len(CARD_LOCS) + 1 + 5
OPTION_EXTRA_DIM = 48
OPTION_DIM = OPTION_BASE_DIM + OPTION_EXTRA_DIM

SINGLE_HEAL_DESC = (10050113 << 20) | 0
ALL_HEAL_DESC = (10050113 << 20) | 1
COMMON_HEAL_DESCS = (SINGLE_HEAL_DESC, ALL_HEAL_DESC, 0)
# "돗테오키 효과" / "두번째 돗테오키 효과" (supports, alchemists), "돗테오키 효과를 발동한다." (battle creamates)
DOTTE_DESCS = ((10050113 << 20) | 4, (10050113 << 20) | 5, (10050111 << 20) | 5)
DAMAGE_DESCS = DOTTE_DESCS + ((10050113 << 20) | 2, (10050113 << 20) | 3)   # + 단일 공격 / 광역 공격
PROFESSOR = 10051332   # 교수(오미치 미야비): +1 per opponent creamate sharing an attribute with our field
SHINOBU = 10051521     # 오오미야 시노부
COUNTER_NO_HEAL = 0xb02     # 불행: cannot be healed (at resolution: a heal chained before a cleanse works)
CLEANSE_TEXT = "상태이상을 전부 해제"   # card text of a status cleanse
CLEANSE_DESC_TEXT = "상태이상 해제"     # ... and of its option
TAUNT_TEXT = "상대는 자신만 통상 공격할 수 있다"
COUNTER_ATTACKS = 0xc01      # normal attacks a battle creamate has left this turn
COUNTER_ALL_ATTACK = 0xc04   # its next normal attack hits every enemy (히로)
# phases of the opponent's turn in which its battle creamates can still deal damage to ours
THREAT_PHASES = (PHASE_DRAW, PHASE_STANDBY, PHASE_MAIN1, PHASE_BATTLE_START, PHASE_BATTLE_STEP, PHASE_DAMAGE,
                 PHASE_DAMAGE_CAL, PHASE_BATTLE)
BEFORE_BATTLE = (PHASE_DRAW, PHASE_STANDBY, PHASE_MAIN1)
PLAN_PHASES = (PHASE_MAIN1, PHASE_BATTLE_START, PHASE_BATTLE_STEP, PHASE_BATTLE)   # our turn look-ahead (tactical)
PLAN_BUFFS = 2   # hand buffs the look-ahead tries
PLAN_TOOLS = 2   # ... and hand gauge supports, extra-attack supports (each)
_PLANS = {}      # look-ahead results by board (decisions of one turn often see the same board)
OPP_HEAL_GUESS = 4          # the opponent's hand is hidden: assume a plain heal of 4 whenever one can be played
SUPPORT_SLOTS = 5
ATTR_LIGHT, ATTR_DARK = 0x10, 0x20   # 해, 달
# Battle creamates whose dotte deals +1 when another own card of these attributes is on the field
# (support zone included): code -> (attributes, base damage of the dotte given the view)
ATTRIBUTE_ENABLERS = {
    10051521: (ATTR_LIGHT | ATTR_DARK,   # 오오미야 시노부: 1, +2 if the opponent has a 해/달 card
               lambda opp_attrs: 1 + (2 if opp_attrs & (ATTR_LIGHT | ATTR_DARK) else 0)),
    10051622: (ATTR_DARK, lambda opp_attrs: 2),   # 마츠바라 호노카
}


_SECTION = re.compile(r"\[([^\]]*)\]")
_BUFF_LINE = re.compile(r"공격력\+(\d+)")


def _sections(desc):
    """[(header, body)] of a card text: "[스킬] ...", "[돗테오키 게이지 2, 추가 경직 1장] ..." and so on."""
    parts = _SECTION.split(desc)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


_GAUGE_COST = re.compile(r"돗테오키 게이지\s*(\d+)")
_STIFF_COST = re.compile(r"추가 경직\s*(\d+)장")   # that many battle creamates in attack position turned stiff
_buff_tables = {}


def stiff_cost(head):
    m = _STIFF_COST.search(head)
    return int(m.group(1)) if m else 0


def upright(cards):
    """Battle creamates that can still pay an "추가 경직" cost (attack position)."""
    return sum(1 for c in cards if not c["pos"] & POS_DEFENSE)


def buff_table(db):
    """[(level, attack +N, all allies?, gauge cost, stiff cost)] of every attack buff a support of the card
    pool can give from the hand (스킬 / 돗테오키 effects): what an opponent may be holding."""
    table = _buff_tables.get(id(db))
    if table is None:
        table = []
        for code in db.creamate_pool():
            card = db[code]
            if not card.is_support:
                continue
            for head, body in _sections(card.desc):
                cost = _GAUGE_COST.match(head)
                if not (head.startswith("스킬") or cost):
                    continue
                for line in body.splitlines():
                    m = _BUFF_LINE.search(line)
                    if m and "아군" in line:
                        table.append((card.level, int(m.group(1)), "아군 전체" in line, int(cost.group(1)) if cost else 0,
                                      stiff_cost(head)))
                        break
        _buff_tables[id(db)] = table
    return table


_tool_tables = {}


def tool_table(db):
    """[(level, "gauge" / "attack", gauge +N (-N: the other side's field creamates + N) / normal attacks +N,
    whose ("one" / "all"; None for gauge), gauge cost, stiff cost)] of the gauge and extra-attack supports
    of the card pool (cardtext gauge_gain / more_attacks)."""
    table = _tool_tables.get(id(db))
    if table is None:
        table = []
        for code in db.creamate_pool():
            card = db[code]
            if not card.is_support:
                continue
            t = text_of(db, code)
            table += [(card.level, "gauge", n, None, cost, stiff) for n, cost, stiff in t.gauge_gain]
            table += [(card.level, "attack", n, who, cost, stiff) for who, n, cost, stiff in t.more_attacks]
        _tool_tables[id(db)] = table
    return table


def _loc_onehot(loc, seq):
    v = [0.0] * len(CARD_LOCS)
    emz = loc == LOCATION_MZONE and seq >= 5
    for i, (l, e) in enumerate(CARD_LOCS):
        if l == loc and (l != LOCATION_MZONE or e == emz):
            v[i] = 1.0
            break
    return v


def _player_feats(view_p, der_p, removed_classes):
    return [
        1.0,   # (was LP / 8000: LP is no game measure - scripts only set it for the BGM and to end the game)
        view_p.get("deck", 0) / 60.0,
        view_p.get("hand", 0) / 10.0,
        der_p["dotte"] / 10.0,
        view_p.get("removed", 0) / 60.0,
        view_p.get("extra", 0) / 20.0,
        view_p.get("extra_facedown", 0) / 20.0,
        view_p.get("summoner_overlay", 0) / 20.0,
        der_p["sanctuary"] / 10.0,
        der_p["to_realist"] / 10.0,
        der_p["starlight"] / 20.0,
        der_p["field_level"] / 20.0,
        max(-10, min(20, der_p["margin"])) / 10.0,
        der_p["battle_count"] / 3.0,
        der_p["hand_battle"] / 10.0,
        1.0 if der_p["can_field_battle"] else 0.0,
        der_p["max_atk"] / 10.0,
        der_p["sum_atk"] / 20.0,
        der_p["attack_counters"] / 5.0,
        der_p["hand_over"] / 5.0,
        1.0 if der_p["hand"] > derived.HAND_CAP else 0.0,
        der_p["killable_levels"] / 10.0,
    ] + [n / 20.0 for n in removed_classes]


def _card_static(db, code):
    card = db.get(code) if code else None
    cls = [0.0] * (len(CLASS_RACES) + 1)
    att = [0.0] * (len(ATTRIBUTES) + 1)
    if card is None:
        cls[-1] = 1.0
        att[-1] = 1.0
        return cls + att + [0.0, 0.0]
    cls[card.class_index] = 1.0
    att[card.attribute_index] = 1.0
    return cls + att + [card.attack / 5.0, card.defense / 10.0]


class _Tactics:
    """Precomputed hints for heal and attack options: how much a heal would really restore, and what
    an attack leaves behind (including how much a heal by the opponent could then give back)."""

    def __init__(self, view, decision, db, der):
        self.view, self.db, self.der = view, db, der
        self.me = view["me"]
        self.battle = ({}, {})
        for c in view["cards"]:
            if c["loc"] == LOCATION_MZONE and c["seq"] < 5 and derived.is_battle_race(db.get(c["code"]) if c["code"]
                                                                                       else None):
                self.battle[c["rel"]][c["seq"]] = c
        self.ctx = decision.context
        self.decision = decision
        self.can_heal = [self._can_heal(rel) for rel in (0, 1)]
        # an opponent battle creamate attacking one of ours right now: (attacker, target)
        self.incoming = None
        att = view.get("attack")
        if att and att["attacker"][0] == 1 and att["target"][0] == 0:
            a, t = self.battle[1].get(att["attacker"][1]), self.battle[0].get(att["target"][1])
            if a is not None and t is not None:
                self.incoming = (a, t)
        # an opponent's dotte on top of the chain that we are answering: (its battle creamate, our target or
        # None for an all-enemy one) - single-target dottes show their target (MSG_BECOME_TARGET)
        self.incoming_dotte = None
        chain = view.get("chain") or ()
        if view["turn_player"] == 1 and chain and chain[-1]["rel"] == 1 and self.incoming is None:
            top = chain[-1]
            b = next((c for c in self.battle[1].values() if c["code"] == top["code"]), None)
            t = text_of(db, top["code"]) if b is not None else None
            if t is not None and t.dotte_hits:
                hit = [self.battle[0].get(seq) for rel, loc, seq in top.get("targets", ())
                       if rel == 0 and loc == LOCATION_MZONE]
                hit = [c for c in hit if c is not None]
                if hit or t.dotte_to_all:
                    self.incoming_dotte = (b, hit[0] if hit else None)
        # what the opponent can still do to our creamates this turn (its battle creamates are known once
        # they stand on the field; buffs from its hidden hand are not counted)
        self.threat_turn = view["turn_player"] == 1 and view.get("phase") in THREAT_PHASES
        self.opp_gauge = der["players"][1]["dotte"]
        self.taunts = [c for c in self.battle[0].values() if c["pos"] & POS_FACEUP and self.db.get(c["code"])
                       and TAUNT_TEXT in self.db[c["code"]].desc]
        hand = [c["code"] for c in view["cards"] if c["rel"] == 0 and c["loc"] == LOCATION_HAND]
        self.hand_cleanses = [h for h in hand if self._is_cleanse_card(h)]
        self.hand_heals = [h for h in hand if self.db.get(h) and self.db[h].is_support
                           and (text_of(db, h).heal_single or text_of(db, h).heal_all)]
        self.slots_used = [sum(1 for c in view["cards"] if c["rel"] == rel and c["loc"] == LOCATION_SZONE
                               and c["seq"] < 5) for rel in (0, 1)]
        self.opp_buff = self._opp_buff_potential()

    def _opp_buff_potential(self):
        """(single +N, all-ally +N): the biggest attack buffs of the card pool the opponent could still play
        from its hidden hand this turn (a free support slot, the card's level within its sanctuary room,
        the gauge cost within the gauge it will have)."""
        p = self.view["players"][1]
        if p.get("hand", 0) <= 0 or self.slots_used[1] >= SUPPORT_SLOTS:
            return 0, 0
        room = 99 if p.get("summoner_code") == 10050110 else self.der["players"][1]["margin"]
        gauge = self._future_gauge(1)
        stand = upright(self.battle[1].values())
        ok = [(n, to_all) for level, n, to_all, cost, stiff in buff_table(self.db)
              if level <= room and cost <= gauge and stiff <= stand]
        return max((n for n, _ in ok), default=0), max((n for n, a in ok if a), default=0)

    def _opp_tool_potential(self):
        """(gauge +N, one more normal attack for one ally, ... for every ally): the best gauge and extra
        attack supports of the card pool the opponent could still play from its hidden hand this turn
        (same limits as _opp_buff_potential). 텐노지 나기사's gain counts our field."""
        p = self.view["players"][1]
        if p.get("hand", 0) <= 0 or self.slots_used[1] >= SUPPORT_SLOTS:
            return 0, False, False
        room = 99 if p.get("summoner_code") == 10050110 else self.der["players"][1]["margin"]
        gauge = self._future_gauge(1)
        stand = upright(self.battle[1].values())
        ours = len(self._field_cards(0))
        g, one, every = 0, False, False
        for level, kind, n, who, cost, stiff in tool_table(self.db):
            if level > room or cost > gauge or stiff > stand:
                continue
            if kind == "gauge":
                g = max(g, n if n > 0 else ours - n)
            elif who == "all":
                every = True
            else:
                one = True
        return g, one, every

    # ---- what a battle creamate's attacks do ------------------------------
    def _class_name(self, c):
        card = self.db.get(c["code"])
        return CLASS_NAMES[CLASS_RACES[card.class_index]] if card is not None and card.class_index < len(CLASS_RACES) \
            else None

    def _attr_name(self, c):
        card = self.db.get(c["code"])
        return ATTRIBUTE_NAMES.get(card.attribute) if card is not None else None

    def start_bonus(self, rel):
        """{seq: (attack +N this turn, +N on its next attack only)} from player rel's "[자신 배틀 페이즈 개시시]"
        effects still to resolve: only on its own turn, while the battle phase has not started
        (_counters_pending: the same step's Kirafan.battleop)."""
        cache = self.__dict__.setdefault("_start_bonus", {})
        if rel in cache:
            return cache[rel]
        out = {}
        if rel == self.view["turn_player"] and self._counters_pending(rel):
            battle = list(self.battle[rel].values())
            hunger = sum(1 for side in (0, 1) for c in self.battle[side].values()
                         if c["counters"].get(COUNTER_HUNGER, 0) > 0)
            for c in battle:
                for rule in text_of(self.db, c["code"]).start:
                    if rule[0] == "allies":
                        for a in battle:
                            if self._class_name(a) in rule[1]:
                                turn, once = out.get(a["seq"], (0, 0))
                                out[a["seq"]] = (turn, once + rule[3]) if rule[2] else (turn + rule[3], once)
                        continue
                    if rule[0] == "self_cond":
                        n = rule[3] if c["def"] >= rule[1] or c["atk"] >= rule[2] else 0
                    elif rule[0] == "self_hand_diff":
                        n = max(0, self.view["players"][1 - rel].get("hand", 0) - self.view["players"][rel].get("hand", 0))
                    else:
                        n = hunger
                    turn, once = out.get(c["seq"], (0, 0))
                    out[c["seq"]] = (turn + n, once)
        cache[rel] = out
        return out

    def attack_of(self, c):
        """(attack for the rest of the turn, + on its next attack only) of battle creamate c."""
        turn, once = self.start_bonus(c["rel"]).get(c["seq"], (0, 0))
        return c["atk"] + turn, once

    def hit_of(self, c, target, atk):
        """Damage one normal attack of c with attack `atk` deals to `target` (not for 절반 all-enemy attackers):
        + a "달 속성 크리에메이트에게 통상 공격시 공격력+1" passive."""
        vs = text_of(self.db, c["code"]).vs_attr_atk
        return max(0, atk + (vs[1] if vs and self._attr_name(target) == vs[0] else 0))

    def attackable(self, target, alive=None):
        """Can an enemy normal attack pick `target`: a taunt on its side keeps the others off, and 봇치 cannot be
        attacked while another of its side stands in the battle zone. alive: that side's battle creamates
        (default: the view's)."""
        side = list(self.battle[target["rel"]].values()) if alive is None else alive
        taunts = [c for c in side if self.is_taunt(c)]
        if taunts and not any(c is target for c in taunts):
            return False
        return not (text_of(self.db, target["code"]).not_attackable and any(c is not target for c in side))

    def dotte_damage(self, t, attacker, target, opt=False, atk_bonus=0):
        """Damage of attacker's dotte effect on `target` (base, field-dependent part, conditional extras; opt:
        with the optional extra cost; atk_bonus: attack added before it, for "공격력 N 이상이면" conditions)."""
        rel = attacker["rel"]
        single, to_all = t.dotte_single, t.dotte_all
        if t.dotte_var:
            all_part, kind, attr, plus = t.dotte_var
            if kind == "attr":
                n = sum(1 for c in self._field_cards(rel) if self._attr_name(c) == attr) + plus
            else:
                n = len(self.battle[1 - rel])
            if all_part:
                to_all += n
            else:
                single += n
        bonus = self._bonus(t, attacker, target, opt, atk_bonus)
        return max(0, to_all + (max(0, single + bonus) if single else bonus))

    def _is_cleanse_card(self, code):
        card = self.db.get(code)
        return card is not None and card.is_support and CLEANSE_TEXT in card.desc

    def _is_cleanse_option(self, code, desc):
        if desc in DOTTE_DESCS:
            return CLEANSE_TEXT in self.db[code].desc.split("[돗테오키", 1)[-1] if self.db.get(code) else False
        return bool(desc) and CLEANSE_DESC_TEXT in self.db.describe(desc)

    def _supports_room(self, n):
        """Can we still put n more supports into play (free support slots and sanctuary room)?"""
        p = self.view["players"][0]
        return self.slots_used[0] + n <= SUPPORT_SLOTS and (p.get("summoner_code") == 10050110
                                                            or self.der["players"][0]["margin"] >= n)

    # ---- the opponent's remaining damage this turn --------------------
    def _attackable(self, target):
        return self.attackable(target)

    def _counters_pending(self, rel):
        """The battle phase has not handed out this turn's attack counters to player rel's battle creamates
        yet: before it, or in its start step before Kirafan.battleop (EVENT_PHASE+PHASE_BATTLE_START, after
        the step's first chain window) - no attack happens in that step, so none has a counter then."""
        phase = self.view.get("phase")
        return phase in BEFORE_BATTLE or (phase == PHASE_BATTLE_START and not any(
            c["counters"].get(COUNTER_ATTACKS, 0) for c in self.battle[rel].values()))

    def _attacks_left(self, b):
        """Normal attacks battle creamate b still has this turn: counter 0xc01, +1 while the battle phase
        has not given it yet (Kirafan.battleop gives every battle creamate one); the attack being resolved
        is still counted on its attacker until its damage."""
        n = b["counters"].get(COUNTER_ATTACKS, 0) + (1 if self._counters_pending(b["rel"]) else 0)
        if self.incoming is not None and b is self.incoming[0]:
            n -= 1
        return max(0, n)

    def _future_gauge(self, rel):
        """Gauge of player rel by the end of its attacks: every battle creamate's first normal attack of the
        turn adds 1 (Kirafan2.AttackDotteop; 3 for 소냐; alchemists sent to the dotte zone on the way are not
        foreseen). A creamate that still has an attack is taken as not having attacked yet, and so is the
        one whose attack is being resolved (its gauge comes at the end of the damage step)."""
        pending = self._counters_pending(rel)
        return self.der["players"][rel]["dotte"] + self.pending_attack_gauge(rel) + sum(
            text_of(self.db, b["code"]).attack_gauge for b in self.battle[rel].values()
            if b["counters"].get(COUNTER_ATTACKS, 0) + (1 if pending else 0) > 0)

    def pending_attack_gauge(self, rel):
        """Gauge the normal attack being resolved still brings player rel at the end of its damage step: its
        attacker's counter is already spent (Kirafan2.battleop, before the damage), taken as its first
        attack when none is left."""
        att = self.view.get("attack")
        if not att or att["attacker"][0] != rel or self.view.get("phase") not in (PHASE_DAMAGE, PHASE_DAMAGE_CAL):
            return 0
        b = self.battle[rel].get(att["attacker"][1])
        if b is None or b["counters"].get(COUNTER_ATTACKS, 0) > 0:
            return 0
        return text_of(self.db, b["code"]).attack_gauge

    def chain_damage(self, target, opt=False):
        """Damage the dottes of the other side still waiting on the chain will deal to `target` (their users
        are stiff already, so threat / plan see them no more)."""
        rel = 1 - target["rel"]
        total = 0
        for ch in self.view.get("chain") or ():
            if ch["rel"] != rel:
                continue
            b = next((c for c in self.battle[rel].values() if c["code"] == ch["code"]), None)
            t = text_of(self.db, ch["code"]) if b is not None else None
            if t is None or not t.dotte_hits:
                continue
            hit = [seq for r, loc, seq in ch.get("targets", ()) if r == target["rel"] and loc == LOCATION_MZONE]
            if t.dotte_to_all or target["seq"] in hit:
                total += self.dotte_damage(t, b, target, opt)
        return total

    @staticmethod
    def _can_dotte(b):
        """A battle creamate's dotte needs it in attack position and free of 속박 (Kirafan6.damcon)."""
        return not b["pos"] & POS_DEFENSE and b["counters"].get(COUNTER_BIND, 0) == 0

    def _normal_hits(self, b, target, cut=0, plus=0, opened=False):
        """(damage of b's next normal attack on `target`, of each one after it) this turn - 0 when it cannot
        attack it; a 절반 all-enemy attacker (Kirafan2.Allbattleop) hits everyone for floor(attack / 2); one
        whose attack gives 나약함 (시마 린) counts one more on its first hit. cut / plus: attack taken off
        (수면) / added; opened: the taunts keeping it off `target` fall first."""
        t = text_of(self.db, b["code"])
        atk, once = self.attack_of(b)
        atk += plus - cut
        if t.half_all:
            return max(0, atk + once) // 2, max(0, atk) // 2
        if opened or b["counters"].get(COUNTER_ALL_ATTACK, 0) > 0 or self.attackable(target):
            weak = 1 if t.attack_weak and not target["counters"].get(COUNTER_WEAK, 0) else 0
            return self.hit_of(b, target, atk + once) + weak, self.hit_of(b, target, atk)
        return 0, 0

    def _taunts_fall(self, target):
        """The taunts that keep the opponent's normal attacks off `target` can all be defeated this turn."""
        taunts = [c for c in self.battle[target["rel"]].values() if self.is_taunt(c) and c is not target]
        return bool(taunts) and all(self.threat(c)[0] >= derived.lethal_hp(c) for c in taunts)

    def _dotte_max(self, target, buff=0, gauge_plus=0):
        """Most the opponent's dotte effects can still add to its damage on `target` this turn with the gauge
        it will have (+ gauge_plus): their damage (base, field-dependent, conditional extras with `buff`
        attack added), 나약함 (HP 1 falls: one damage more), the attack they give their user for its normal
        attacks left, and the normal attacks they add (paying their extra stiff). Each once, except a
        creamate that its normal attack turns upright again (혼다 타마키: once more per attack)."""
        entries = []
        standing = [f for f in self.battle[1].values() if not f["pos"] & POS_DEFENSE]
        weak_gain = 0 if target["counters"].get(COUNTER_WEAK, 0) else 1
        for b in self.battle[1].values():
            t = text_of(self.db, b["code"])
            if not t.dotte_hits or not self._can_dotte(b):
                continue
            others = sum(1 for f in standing if f is not b)
            if t.dotte_stiff > others:
                continue
            base = self.dotte_damage(t, b, target, opt=True, atk_bonus=buff) + (weak_gain if t.dotte_weak else 0)
            value = base
            n = self._attacks_left(b)
            first, later = self._normal_hits(b, target)
            up = t.dotte_self_atk + (len(self.battle[0]) if t.dotte_self_atk_opp else 0)
            if up and n:
                first2, later2 = self._normal_hits(b, target, plus=up)
                value += (first2 - first) + (n - 1) * (later2 - later)
            more = t.dotte_more_attacks
            if more and more[0] + t.dotte_stiff <= others:
                per = [self._normal_hits(f, target)[1 if self._attacks_left(f) else 0] for f in self.battle[1].values()]
                own = later if n else first
                value += more[2] * (own if more[1] == "self" else (max(per, default=0) if more[1] == "one" else sum(per)))
            entries.append((t.dotte, value))
            if t.unstiff_after_attack:
                entries += [(t.dotte, base)] * (n + (more[2] if more and more[1] == "self" else 0))
        entries = sorted(entries, key=lambda e: -e[1])[:10]
        gauge = self._future_gauge(1) + gauge_plus
        best = 0
        for mask in range(1 << len(entries)):
            cost = sum(d[0] for i, d in enumerate(entries) if mask >> i & 1)
            if cost <= gauge:
                best = max(best, sum(d[1] for i, d in enumerate(entries) if mask >> i & 1))
        return best

    def threat(self, target, sleep_all=False, sleep_attacker=False, buff=False):
        """(total, normal attacks after the current one, dotte) damage the opponent can still deal to
        `target` this turn: the attack being resolved (if on it), every normal attack its battle creamates
        have left (all-enemy attacks too; a taunt keeps the others off our other creamates), and its dotte
        effects within the gauge it will have. sleep_*: a new 수면 (-1) on all of them / the attacker.
        buff: also the best single card it could still play from the hidden hand: an attack buff (opp_buff:
        +N on one attack, or +N on every attacker's next attack), a gauge support (more dottes) or one that
        adds normal attacks (_opp_tool_potential) - before declaring its next attacks."""
        if not self.threat_turn:
            return 0, 0, 0
        memo = self.__dict__.setdefault("_threats", {})
        key = (id(target), sleep_all, sleep_attacker, buff)
        if key not in memo:
            memo[key] = self._threat(target, sleep_all, sleep_attacker, buff)
        return memo[key]

    def _threat(self, target, sleep_all, sleep_attacker, buff):
        if buff:
            total, rest, dotte = self.threat(target, sleep_all, sleep_attacker)
            single, to_all = self.opp_buff
            # attacks still to be declared only: while attacking, the attacker cannot play supports
            hitters = sum(1 for b in self.battle[1].values() if self._attacks_left(b) and self._normal_hits(b, target)[0])
            extra = max(single, to_all * hitters) if hitters else 0
            gain = self._dotte_max(target, single) - dotte   # one card: on the attacks, or for a dotte's condition
            g, one, every = self._opp_tool_potential()
            if g:
                gain = max(gain, self._dotte_max(target, gauge_plus=g) - dotte)
            per = [self._normal_hits(b, target)[1 if self._attacks_left(b) else 0] for b in self.battle[1].values()]
            if one:
                extra = max(extra, max(per, default=0))
            if every:
                extra = max(extra, sum(per))
            if gain > extra:
                return total + gain, rest, dotte + gain
            return total + extra, rest + extra, dotte
        cur = None
        now = 0
        if self.incoming is not None:
            cur = self.incoming[0]
            asleep = cur["counters"].get(COUNTER_SLEEP, 0) > 0
            cut = 1 if (sleep_all or sleep_attacker) and not asleep else 0
            if text_of(self.db, cur["code"]).half_all:
                now = max(0, cur["atk"] - cut) // 2
            elif self.incoming[1] is target:
                now = self.hit_of(cur, target, cur["atk"] - cut)
        rest = 0
        opened = not self.attackable(target) and self._taunts_fall(target)
        for b in self.battle[1].values():
            n = self._attacks_left(b)
            if not n:
                continue
            asleep = b["counters"].get(COUNTER_SLEEP, 0) > 0
            cut = 1 if (sleep_all or (sleep_attacker and b is cur)) and not asleep else 0
            first, later = self._normal_hits(b, target, cut=cut, opened=opened)
            rest += first + (n - 1) * later
        dotte = self._dotte_max(target) + self.chain_damage(target, opt=True)
        return now + rest + dotte, rest, dotte

    def lethal_this_turn(self, c):
        return self.threat_turn and self.threat(c)[0] >= derived.lethal_hp(c)

    def _turn_target(self, o):
        """Our creamate this option is about against the opponent's turn: the one being attacked, or the
        ally picked for a heal."""
        if self.incoming is not None:
            return self.incoming[1]
        eff = self._effect_of(o)
        if eff and eff[2] is not None and eff[2]["rel"] == 0 and self._heal_of(eff[0], eff[1])[0]:
            return eff[2]
        return None

    def _turn_feats(self, o, target):
        """([threat / 10, defeated this turn, this option saves it for the turn, cannot save it for the
        turn], [threat with the opponent's possible hidden attack buff / 10, defeated then, this option
        saves it even then, saves it only if the opponent holds no such buff]) - saves / cannot save only
        for an option that heals it or puts 수면 on the attackers."""
        total, _, _ = self.threat(target)
        total_b, _, _ = self.threat(target, buff=True)
        lethal = derived.lethal_hp(target)
        dies, dies_b = total >= lethal, total_b >= lethal
        out = [total / 10.0, 1.0 if dies else 0.0, 0.0, 0.0]
        out_b = [total_b / 10.0, 1.0 if dies_b else 0.0, 0.0, 0.0]
        eff = self._effect_of(o)
        if eff is None:
            return out, out_b
        code, desc, picked = eff
        hp, acts = target["def"], False
        amount, to_all, overheal = self._heal_of(code, desc)
        if amount and (picked is None or picked is target):
            hp += self._heal_on(target, amount, overheal)
            acts = True
        status = text_of(self.db, code).enemy_status
        sleep_all = sleep_att = False
        if status and status[0] == COUNTER_SLEEP and desc not in COMMON_HEAL_DESCS:
            acts = True
            sleep_all = status[1]
            sleep_att = not status[1] and (picked is None or (self.incoming is not None and picked is self.incoming[0]))
        if not acts:
            return out, out_b
        lethal_after = derived.lethal_hp(dict(target, **{"def": hp}))
        after = self.threat(target, sleep_all, sleep_att)[0] >= lethal_after
        after_b = self.threat(target, sleep_all, sleep_att, buff=True)[0] >= lethal_after
        out[2:4] = [1.0 if dies and not after else 0.0, 1.0 if dies and after else 0.0]
        out_b[2:4] = [1.0 if dies_b and not after_b else 0.0, 1.0 if not after and after_b else 0.0]
        return out, out_b

    # ---- attack buffs on our turn ----------------------------------------
    def _buff_of(self, code, desc):
        """(attack +N, all allies?) of the attack buff a support's option gives, from the card text section
        of that option (돗테오키 게이지 / 스킬)."""
        card = self.db.get(code)
        if card is None or not card.is_support or desc in COMMON_HEAL_DESCS:
            return 0, False
        if desc in DOTTE_DESCS:
            want = "돗테오키 게이지"
        elif desc and "공격력" in self.db.describe(desc):
            want = "스킬"
        else:
            return 0, False
        for head, body in _sections(card.desc):
            if not head.startswith(want):
                continue
            for line in body.splitlines():
                m = _BUFF_LINE.search(line)
                if m and "아군" in line:
                    return int(m.group(1)), "아군 전체" in line
        return 0, False

    def is_taunt(self, c):
        return bool(c["pos"] & POS_FACEUP) and self.db.get(c["code"]) is not None and TAUNT_TEXT in self.db[c["code"]].desc

    def hand_buffs(self):
        """[(code, attack +N, all allies?, one attack only?, level, gauge cost, stiff cost)] of the attack
        buffs in our hand (each copy; the PLAN_BUFFS biggest)."""
        out = []
        for c in self.view["cards"]:
            if c["rel"] != 0 or c["loc"] != LOCATION_HAND:
                continue
            card = self.db.get(c["code"])
            if card is None or not card.is_support:
                continue
            best = None
            for head, body in _sections(card.desc):
                cost = _GAUGE_COST.match(head)
                if not (head.startswith("스킬") or cost):
                    continue
                for line in body.splitlines():
                    m = _BUFF_LINE.search(line)
                    if m and "아군" in line:
                        entry = (c["code"], int(m.group(1)), "아군 전체" in line, "1번만" in line, card.level,
                                 int(cost.group(1)) if cost else 0, stiff_cost(head))
                        if best is None or entry[1] > best[1]:
                            best = entry
                        break
            if best:
                out.append(best)
        out.sort(key=lambda b: (-b[1], b[5], b[4]))
        return out[:PLAN_BUFFS]

    def hand_tools(self):
        """[(code, "gauge" / "attack", gauge +N / normal attacks +N, whose ("one" / "all"; None for gauge),
        level, gauge cost, stiff cost)] of the gauge and extra-attack supports in our hand (one use per copy,
        copies already counted as hand_buffs left out; the PLAN_TOOLS best of each kind). 텐노지 나기사's gain
        counts the opponent's field now."""
        opp_field = len(self._field_cards(1))
        taken = collections.Counter(b[0] for b in self.hand_buffs())
        gauge, attack = [], []
        for c in self.view["cards"]:
            if c["rel"] != 0 or c["loc"] != LOCATION_HAND:
                continue
            card = self.db.get(c["code"])
            if card is None or not card.is_support:
                continue
            if taken[c["code"]]:
                taken[c["code"]] -= 1
                continue
            t = text_of(self.db, c["code"])
            if t.gauge_gain:
                n, cost, stiff = max(t.gauge_gain, key=lambda g: g[0] if g[0] > 0 else opp_field - g[0])
                gauge.append((c["code"], "gauge", n if n > 0 else opp_field - n, None, card.level, cost, stiff))
            elif t.more_attacks:
                who, n, cost, stiff = t.more_attacks[0]
                attack.append((c["code"], "attack", n, who, card.level, cost, stiff))
        gauge.sort(key=lambda g: (-g[2], g[5], g[4]))
        attack.sort(key=lambda g: (g[3] != "all", g[5], g[4]))
        return gauge[:PLAN_TOOLS] + attack[:PLAN_TOOLS]

    def defense_cards(self):
        """[(hand card index, "heal" / "sleep", amount, all allies?, overheal, level, gauge cost, 수면 with it,
        card code)]
        the answers our hand holds against the opponent's attacks (tactical.Defense): common heals, dotte
        heals (with the 수면 of the same effect, 야마구치 키사라기) and 수면 effects."""
        out = []
        cards = [c for c in self.view["cards"] if c["rel"] == 0 and c["loc"] == LOCATION_HAND]
        for h, c in enumerate(cards):
            card = self.db.get(c["code"])
            if card is None or not card.is_support:
                continue
            t = text_of(self.db, c["code"])
            status = t.enemy_status
            sleep = ("all" if status[1] else "one") if status and status[0] == COUNTER_SLEEP else None
            code = c["code"]
            if t.heal_single:
                out.append((h, "heal", t.heal_single, False, t.overheal, card.level, 0, None, code))
            if t.heal_all:
                out.append((h, "heal", t.heal_all, True, t.overheal, card.level, 0, None, code))
            if t.dotte_heal_single or t.dotte_heal_all:
                out.append((h, "heal", t.dotte_heal_all or t.dotte_heal_single, bool(t.dotte_heal_all), t.overheal,
                            card.level, t.dotte, sleep, code))
            elif sleep:
                out.append((h, "sleep", 0, sleep == "all", False, card.level, 0, None, code))
        return out

    def _my_kill_dotte(self):
        """One of our battle creamates can defeat an enemy battle creamate with its dotte right now."""
        gauge = self.der["players"][0]["dotte"]
        for b in self.battle[0].values():
            t = text_of(self.db, b["code"])
            if not t.dotte_hits or t.dotte > gauge or not self._can_dotte(b):
                continue
            for e in self.battle[1].values():
                dmg = self.dotte_damage(t, b, e)
                if dmg >= derived.lethal_hp(e) - (1 if t.dotte_weak and not e["counters"].get(COUNTER_WEAK, 0) else 0):
                    return True
        return False

    def _buff_feats(self, o):
        """[attack buff +N / 3, no buffed ally has a normal attack left this turn, the buff lets a normal
        attack defeat an enemy it would not have, one of our dottes can defeat an enemy first (spend the
        opponent's heals on that, then buff and attack)] - on our turn, for attack buffs."""
        eff = self._effect_of(o)
        if eff is None or self.view["turn_player"] != 0:
            return [0.0] * 4
        code, desc, picked = eff
        amount, to_all = self._buff_of(code, desc)
        if not amount:
            return [0.0] * 4
        allies = list(self.battle[0].values())
        targets = [picked] if picked is not None and picked["rel"] == 0 else allies
        ready = [a for a in targets if self._attacks_left(a) > 0]
        foes = list(self.battle[1].values())
        enemy_taunts = [f for f in foes if f["pos"] & POS_FACEUP and self.db.get(f["code"])
                        and TAUNT_TEXT in self.db[f["code"]].desc]
        hittable = enemy_taunts or foes
        kill = any(a["atk"] < derived.lethal_hp(f) <= a["atk"] + amount for a in ready for f in hittable)
        return [amount / 3.0, 0.0 if ready else 1.0, 1.0 if kill else 0.0, 1.0 if self._my_kill_dotte() else 0.0]

    # ---- look-ahead over the rest of our turn (model.tactical) ---------
    def _opp_heals(self):
        """Heal cards the opponent is taken to hold this turn (tactical.OPP_HEALS_MAX at most). Whether it
        can play them (free slots, sanctuary room - which our kills free up) the look-ahead decides."""
        from .tactical import OPP_HEALS_MAX
        p = self.view["players"][1]
        if self.slots_used[1] >= SUPPORT_SLOTS:
            return 0
        return min(OPP_HEALS_MAX, p.get("hand", 0))

    def plan(self):
        """(Plan, start state with the opponent's heals, best (score, kills) then, best kills if it cannot
        heal), or None outside our turn's main 1 / battle phase."""
        if hasattr(self, "_plan"):
            return self._plan
        self._plan = None
        if self.view["turn_player"] != 0 or self.view.get("phase") not in PLAN_PHASES \
                or not self.battle[0] or not self.battle[1]:
            return None
        from .tactical import Plan
        p0 = self.der["players"][0]
        key = (tuple((c["seq"], c["code"], c["atk"], c["def"], self._attacks_left(c), tuple(sorted(c["counters"].items())))
                     for side in (0, 1) for c in self.battle[side].values()),
               p0["dotte"], p0["margin"], self.slots_used[0], tuple(self.hand_buffs()), tuple(self.hand_tools()),
               self._opp_heals(), self._counters_pending(0), self.view["players"][1].get("hand", 0),
               self.view["players"][0].get("hand", 0))
        hit = _PLANS.get(key)
        if hit is None:
            p = Plan(self)
            s = p.start(self._opp_heals())
            hit = (p, s, p.best(s), p.best(p.start(0)))
            if len(_PLANS) >= 64:
                _PLANS.clear()
            _PLANS[key] = hit
        self._plan = hit
        return self._plan

    def _plan_action_value(self, o):
        """(score, kills) of the rest of our turn if this option is taken (best follow-up), or None when the
        option is no planned action."""
        pl = self.plan()
        if pl is None or o.card is None and o.kind not in (K_IDLE_TO_BP, K_IDLE_TO_EP, K_BATTLE_TO_EP,
                                                           K_BATTLE_TO_M2):
            return None
        p, s, best, _ = pl
        if o.kind == K_IDLE_TO_BP:
            return best
        if o.kind in (K_IDLE_TO_EP, K_BATTLE_TO_EP, K_BATTLE_TO_M2):
            return 0, 0
        mine = {c["seq"]: i for i, c in enumerate(p.mine)}
        foes = {c["seq"]: i for i, c in enumerate(p.foes)}
        code, con, loc, seq = o.card
        rel = 0 if con == self.me else 1
        alive = [e for e in range(len(p.foes)) if s[0][e] > 0]

        def top(match):
            vals = [p.value(s, a) for a in p.actions(s) if match(a)]
            return max(vals) if vals else None

        def buff_index(card_code):
            return next((k for k, b in enumerate(p.buffs) if b[0] == card_code), None)

        if o.kind == K_BATTLE_ATTACK and rel == 0 and loc == LOCATION_MZONE and seq in mine:
            a = mine[seq]
            return top(lambda act: act[0] == "attack" and act[1] == a and act[3] is None)
        if o.kind in ACTIVATE_KINDS and rel == 0 and loc == LOCATION_MZONE and seq in mine and o.desc in DAMAGE_DESCS:
            b = mine[seq]
            return top(lambda act: act[0] == "dotte" and act[1] == b)
        if o.kind in ACTIVATE_KINDS and rel == 0 and loc == LOCATION_HAND and self._buff_of(code, o.desc)[0]:
            k = buff_index(code)
            return None if k is None else top(lambda act: act[0] == "attack" and act[3] == k)
        if o.kind in ACTIVATE_KINDS and rel == 0 and loc == LOCATION_HAND and o.desc not in COMMON_HEAL_DESCS \
                and any(t[0] == code for t in p.tools):
            return top(lambda act: act[0] == "tool" and p.tools[act[1]][0] == code)
        if o.kind == K_CARD_PICK and self.ctx and self.ctx[1] and loc == LOCATION_MZONE:
            kind, ccard, cdesc = self.ctx
            if kind == K_BATTLE_ATTACK and rel == 1 and seq in foes and ccard[3] in mine:
                a, e = mine[ccard[3]], p.same_as(s, foes[seq])
                anyone = p.half[a] or s[14][a]   # (it hits every enemy whichever it picks)
                return top(lambda act: act[0] == "attack" and act[1] == a and (anyone or act[2] == e) and act[3] is None)
            if kind in ACTIVATE_KINDS and rel == 1 and seq in foes and ccard[2] == LOCATION_MZONE \
                    and ccard[3] in mine and cdesc in DAMAGE_DESCS:
                b, e = mine[ccard[3]], p.same_as(s, foes[seq])
                return top(lambda act: act[0] == "dotte" and act[1] == b and act[2] == e)
            if kind in ACTIVATE_KINDS and rel == 0 and seq in mine and self._buff_of(ccard[0], cdesc)[0]:
                k, a = buff_index(ccard[0]), mine[seq]
                return None if k is None else top(lambda act: act[0] == "attack" and act[1] == a and act[3] == k)
        return None

    def _plan_may_use(self, o):
        """False when _plan_action_value(o) is None whatever the plan (no search needed for it)."""
        if o.card is None:
            return o.kind in (K_IDLE_TO_BP, K_IDLE_TO_EP, K_BATTLE_TO_EP, K_BATTLE_TO_M2)
        code, con, loc, seq = o.card
        rel = 0 if con == self.me else 1
        if o.kind == K_BATTLE_ATTACK:
            return rel == 0 and loc == LOCATION_MZONE
        if o.kind in ACTIVATE_KINDS:
            if rel != 0:
                return False
            if loc == LOCATION_MZONE:
                return o.desc in DAMAGE_DESCS
            return loc == LOCATION_HAND and (bool(self._buff_of(code, o.desc)[0]) or (
                o.desc not in COMMON_HEAL_DESCS and any(t[0] == code for t in self.hand_tools())))
        if o.kind == K_CARD_PICK:
            return bool(self.ctx and self.ctx[1]) and loc == LOCATION_MZONE
        return False

    def _plan_needed(self):
        """Does any option of this decision use the plan? (searching costs the most of all features)"""
        if not hasattr(self, "_plan_need"):
            self._plan_need = any(self._plan_may_use(o) for o in self.decision.options)
        return self._plan_need

    def _plan_feats(self, o):
        """[this option is the best plan's next step, best plan's kills / 3 (the opponent healing what it
        can), its kills if the opponent could not heal / 3, kills with this option first / 3, kills it
        gives up against the best plan / 3, the option is a planned action]"""
        if not self._plan_needed():
            return [0.0] * 6
        pl = self.plan()
        v = self._plan_action_value(o) if pl is not None else None
        if v is None:
            return [0.0] * 6
        _, _, best, free = pl
        return [1.0 if v[0] >= best[0] - 1e-6 else 0.0, best[1] / 3.0, free[1] / 3.0, v[1] / 3.0,
                max(0, best[1] - v[1]) / 3.0, 1.0]

    # ---- look-ahead over the rest of the opponent's turn (tactical.Defense) ----
    def defense(self):
        """((Defense, start state, the attack / dotte being resolved as its move, (loss, lost) if we pass),
        the same with the opponent's hidden buff) while an attack of the opponent's is on one of ours or
        one of its dottes is on the chain; else None."""
        if hasattr(self, "_defense"):
            return self._defense
        self._defense = None
        what = self.incoming or self.incoming_dotte
        if what is None or not self.threat_turn:
            return None
        from .tactical import Defense
        key = ("def", tuple((side, c["seq"], c["code"], c["atk"], c["def"], self._attacks_left(c), c["pos"],
                             tuple(sorted(c["counters"].items())))
                            for side in (0, 1) for c in self.battle[side].values()),
               self.der["players"][0]["dotte"], self.der["players"][1]["dotte"], self.der["players"][0]["margin"],
               self.slots_used[0], tuple(sorted(c["code"] for c in self.view["cards"]
                                                if c["rel"] == 0 and c["loc"] == LOCATION_HAND)),
               self.opp_buff, self.incoming is not None, what[0]["seq"], what[1]["seq"] if what[1] else None)
        hit = _PLANS.get(key)
        if hit is None:
            out = []
            for hidden in (False, True):
                d = Defense(self, hidden_buff=hidden, resolving=None if self.incoming else what[0])
                s = d.start()
                b = next(i for i, c in enumerate(d.foes) if c is what[0])
                t = next((i for i, c in enumerate(d.mine) if c is what[1]), None)
                if self.incoming is not None:
                    # the attack being resolved is still to land: count it back on its attacker (taken as its
                    # first attack, whose gauge comes at the end of the damage step)
                    s = s[:7] + (s[7][:b] + (s[7][b] + 1,) + s[7][b + 1:], s[8][:b] + (True,) + s[8][b + 1:]) + s[9:]
                    move = ("attack", b, t, False)
                else:
                    # the dotte on the chain: its gauge is paid already, its damage still to come
                    s = s[:10] + (s[10] + d.dottes[b][0],) + s[11:]
                    move = ("dotte", b, t, None)
                out.append((d, s, move, d.exchange(s, move, ((), {}, False, None))))
            hit = tuple(out)
            if len(_PLANS) >= 64:
                _PLANS.clear()
            _PLANS[key] = hit
        self._defense = hit
        return hit

    def _defense_value(self, o, d, s, move):
        """(loss, lost) of this option as our answer to the attack, or None when it maps to no answer."""
        if o.kind == K_CHAIN_PASS:
            return d.exchange(s, move, ((), {}, False, None))
        eff = self._effect_of(o)
        if eff is None:
            return None
        code, desc, picked = eff
        cleanse = self._is_cleanse_option(code, desc)
        heal = self._heal_of(code, desc)[0] > 0
        status = text_of(self.db, code).enemy_status
        sleep = bool(status) and status[0] == COUNTER_SLEEP and desc not in COMMON_HEAL_DESCS
        target = None
        if picked is not None and picked["rel"] == 0:
            target = next((i for i, c in enumerate(d.mine) if c["seq"] == picked["seq"]), None)
        vals = []
        for a in d.answers(s, move):
            used = a[0]
            if cleanse:
                if "cleanse" in used:
                    vals.append(d.exchange(s, move, a))
                continue
            if not used or used[0] == "cleanse" or d.cards[used[0]][8] != code:
                continue
            kind = d.cards[used[0]][1]
            if (heal and kind != "heal") or (sleep and not heal and kind != "sleep"):
                continue
            if target is not None and target not in a[1]:
                continue
            vals.append(d.exchange(s, move, a))
        return min(vals, key=lambda v: v[0]) if vals else None

    def _defense_feats(self, o):
        """[this option is our best answer to the attack, our creamates lost with it this turn / 3, lost if
        we pass / 3, lost with the best answer / 3, same as the second with the opponent's hidden attack
        buff / 3, the option maps to an answer]"""
        dd = self.defense()
        if dd is None:
            return [0.0] * 6
        (d, s, move, passed), (db_, sb, mb, _) = dd
        v = self._defense_value(o, d, s, move)
        if v is None:
            return [0.0] * 6
        best = min((d.exchange(s, move, a) for a in d.answers(s, move)), key=lambda x: x[0])
        vb = self._defense_value(o, db_, sb, mb) or v
        return [1.0 if v[0] <= best[0] + 1e-6 else 0.0, v[1] / 3.0, passed[1] / 3.0, best[1] / 3.0, vb[1] / 3.0, 1.0]

    # ---- two supports in one chain (resolved last-in first-out) ---------
    def _combo_feats(self, o):
        """[a heal on an 불행 ally that works if a cleanse from the hand is chained after it, a cleanse that
        lets a heal work (1: our heal is already on the chain, 0.5: one is in the hand)]"""
        eff = self._effect_of(o)
        if eff is None:
            return [0.0, 0.0]
        code, desc, picked = eff
        allies = list(self.battle[0].values())
        unhappy = [c for c in allies if c["counters"].get(COUNTER_NO_HEAL, 0) > 0]
        if not unhappy:
            return [0.0, 0.0]
        amount, to_all, _ = self._heal_of(code, desc)
        if amount and (picked is None or (picked["rel"] == 0 and any(c is picked for c in unhappy))):
            cleanses = list(self.hand_cleanses)
            from_hand = o.kind in ACTIVATE_KINDS and o.card and o.card[2] == LOCATION_HAND
            if from_hand and code in cleanses:
                cleanses.remove(code)
            if cleanses and self._supports_room(2 if from_hand else 1):
                return [1.0, 0.0]
            return [0.0, 0.0]
        if self._is_cleanse_option(code, desc):
            on_chain = any(ch["rel"] == 0 and self.db.get(ch["code"]) and self.db[ch["code"]].is_support
                           and (text_of(self.db, ch["code"]).heal_single or text_of(self.db, ch["code"]).heal_all)
                           for ch in self.view.get("chain", ()))
            heals = [h for h in self.hand_heals if h != code]
            return [0.0, 1.0 if on_chain else (0.5 if heals and self._supports_room(2) else 0.0)]
        return [0.0, 0.0]

    def _can_heal(self, rel):
        """Could this player still play a heal from the hand (hand, free support slot, sanctuary room)?"""
        p = self.view["players"][rel]
        if p.get("hand", 0) <= 0:
            return False
        used = sum(1 for c in self.view["cards"] if c["rel"] == rel and c["loc"] == LOCATION_SZONE and c["seq"] < 5)
        if used >= SUPPORT_SLOTS:
            return False
        return p.get("summoner_code") == 10050110 or self.der["players"][rel]["margin"] >= 1

    @staticmethod
    def _missing(c):
        return max(0, c["base_def"] - c["def"])

    def _heal_on(self, c, amount, overheal):
        if c["counters"].get(COUNTER_NO_HEAL, 0) > 0:
            return 0
        return amount if overheal else min(amount, self._missing(c))

    def _heal_of(self, code, desc):
        """(amount, all allies?, overheal) of a heal effect, from the card text: the common heals, or
        the heal inside a dotte effect."""
        t = text_of(self.db, code)
        if desc == ALL_HEAL_DESC:
            return t.heal_all, True, t.overheal
        if desc in (SINGLE_HEAL_DESC, 0):
            return t.heal_single, False, t.overheal
        if desc in DOTTE_DESCS:
            if t.dotte_heal_all:
                return t.dotte_heal_all, True, t.overheal
            return t.dotte_heal_single, False, t.overheal
        return 0, False, False

    def _opp_heal_after(self, target, hp_after, killed):
        if killed or not self.can_heal[target["rel"]] or target["counters"].get(COUNTER_NO_HEAL, 0) > 0:
            return 0
        return min(OPP_HEAL_GUESS, max(0, target["base_def"] - hp_after))

    def _hit(self, atk, target):
        hp = target["def"]
        killed = atk >= derived.lethal_hp(target)
        hp_after = 0 if killed else max(0, hp - atk)
        return min(atk, hp), hp_after, killed, max(0, atk - hp), self._opp_heal_after(target, hp_after, killed)

    def _field_attrs(self, rel, skip=None):
        out = 0
        for c in self.view["cards"]:
            if c is skip or c["rel"] != rel or not (c["pos"] & POS_FACEUP) or not c["code"]:
                continue
            if (c["loc"] == LOCATION_MZONE and c["seq"] < 5) or c["loc"] == LOCATION_SZONE:
                card = self.db.get(c["code"])
                out |= card.attribute if card else 0
        return out

    def enables_kill(self, code):
        """Placing this support only for its attribute is worth it: on our turn it completes the
        attribute condition of a ready dotte (nothing else on the field or in hand does), and that +1
        damage is exactly what defeats an opponent battle creamate."""
        card = self.db.get(code)
        if card is None or self.view["turn_player"] != 0 or self.der["players"][0]["dotte"] < 2:
            return False
        hand = [c["code"] for c in self.view["cards"] if c["rel"] == 0 and c["loc"] == LOCATION_HAND]
        if code in hand:
            hand.remove(code)
        opp_attrs = self._field_attrs(1)
        for b in self.battle[0].values():
            rule = ATTRIBUTE_ENABLERS.get(b["code"])
            if rule is None or not (card.attribute & rule[0]) or b["pos"] & POS_DEFENSE:
                continue
            if self._field_attrs(0, skip=b) & rule[0]:
                continue   # already satisfied
            if any(self.db.get(h) and self.db[h].attribute & rule[0] for h in hand):
                continue   # another hand card can do it
            base = rule[1](opp_attrs)
            if any(base < derived.lethal_hp(t) <= base + 1 for t in self.battle[1].values()):
                return True
        return False

    def useless_heals(self, options, extras):
        """Options to treat like no-ops: a common heal that restores nothing (unless placing the support
        enables a kill, see enables_kill), and its target at full HP while another can be healed. Heals
        inside a dotte effect never are: the rest of the effect (e.g. sleep) can still matter."""
        out = [False] * len(options)
        ctx_common = bool(self.ctx) and self.ctx[2] in COMMON_HEAL_DESCS
        best_target = max((e[1] for o, e in zip(options, extras) if o.kind == K_CARD_PICK and e[0]), default=0.0)
        for i, (o, e) in enumerate(zip(options, extras)):
            if not e[0] or e[1] > 0 or e[24] > 0:   # (a heal on 불행 that a later cleanse makes work)
                continue
            if o.kind in ACTIVATE_KINDS and o.desc in COMMON_HEAL_DESCS:
                out[i] = not self.enables_kill(o.card[0])
            elif o.kind == K_CARD_PICK and ctx_common:
                out[i] = best_target > 0
        return out

    def option(self, o):
        """[is heal, heal restored, heal wasted, damage, HP left, defeats, overkill, opponent heal back,
        incoming attack defeats our creamate, its damage, this option saves it, is not needed (it survives
        anyway), cannot save it, share of the targets that already have the status this option gives,
        dotte damage x6 (_dotte_damage_feats),
        against the whole rest of the opponent's turn: threat on our creamate, it is defeated this turn,
        this option saves it for the turn, cannot save it for the turn,
        heal on an 불행 ally that works with a cleanse chained after, cleanse that lets a heal work,
        normal attacks still to come on it, dotte damage still to come,
        attack buff x4 (_buff_feats: amount, wasted, enables a kill, a killing dotte can go first),
        against the rest of the turn with the opponent's possible hidden attack buff: threat, defeated,
        this option saves it even then, saves it only if the opponent holds no buff,
        look-ahead over the rest of our turn x6 (_plan_feats),
        look-ahead over the rest of the opponent's turn while it attacks x6 (_defense_feats)]"""
        out = self._option_core(o)
        if self.incoming is not None:
            attacker, target = self.incoming
            lethal = attacker["atk"] >= derived.lethal_hp(target)
            out[8:10] = [1.0 if lethal else 0.0, attacker["atk"] / 5.0]
            still = self._incoming_after(o)
            if still is not None:
                out[10:13] = [1.0 if lethal and not still else 0.0, 0.0 if lethal else 1.0,
                              1.0 if lethal and still else 0.0]
        out[13] = self._status_dup(o)
        out[14:20] = self._dotte_damage_feats(o)
        target = self._turn_target(o) if self.threat_turn else None
        if target is not None:
            out[20:24], out[32:36] = self._turn_feats(o, target)
            _, rest, dotte = self.threat(target)
            out[26:28] = [rest / 10.0, dotte / 10.0]
        out[24:26] = self._combo_feats(o)
        out[28:32] = self._buff_feats(o)
        out[36:42] = self._plan_feats(o)
        out[42:48] = self._defense_feats(o)
        return out

    # ---- dotte effect damage ------------------------------------------
    def _field_cards(self, rel):
        return [c for c in self.view["cards"] if c["rel"] == rel and (c["pos"] & POS_FACEUP) and c["code"]
                and ((c["loc"] == LOCATION_MZONE and c["seq"] < 5) or c["loc"] == LOCATION_SZONE)]

    def _bonus(self, t, attacker, target, opt, atk_bonus=0):
        """Extra damage of a dotte effect on `target` (rules read from the card text) from the attacker's side
        (either player); opt: also pay the optional extra cost (extra stiff / gauge); atk_bonus: attack the
        attacker gets before it."""
        extra = 0
        rel = attacker["rel"]
        mine = self._field_cards(rel)
        battle = list(self.battle[rel].values())
        for name, g in t.dotte_rules:
            v = [int(x) if x.isdigit() else x for x in g]
            if name == "my_battle_ge" and len(battle) >= v[0]:
                extra += v[1]
            elif name == "my_field_ge" and len(mine) >= v[0]:
                extra += v[1]
            elif name == "target_status" and target["counters"].get(STATUS_COUNTERS[v[0]], 0) > 0:
                extra += v[1]
            elif name == "target_class":
                card = self.db.get(target["code"])
                if card is not None and card.class_index < len(CLASS_RACES) \
                        and CLASS_NAMES[CLASS_RACES[card.class_index]] == v[0]:
                    extra += v[1]
            elif name == "target_attr":
                card = self.db.get(target["code"])
                if card is not None and ATTRIBUTE_NAMES.get(card.attribute) == v[0]:
                    extra += v[1]
            elif name == "opp_hand_ge" and self.view["players"][1 - rel].get("hand", 0) >= v[0]:
                extra += v[1]
            elif name == "no_mage" and not any(self.db.get(c["code"]) and self.db[c["code"]].class_index == 1
                                               for c in battle):
                extra += v[0]
            elif name == "no_support" and not any(c["loc"] == LOCATION_SZONE for c in mine):
                extra += v[0]
            elif name == "other_attr" and any(c is not attacker and self.db.get(c["code"])
                                              and ATTRIBUTE_NAMES.get(self.db[c["code"]].attribute) == v[0]
                                              for c in mine):
                extra += v[1]
            elif name == "self_hp_le_minus" and attacker["def"] <= v[0]:
                extra -= v[1]
            elif name == "self_hp_ge" and attacker["def"] >= v[0]:
                extra += v[1]
            elif name == "self_hp_or_atk" and (attacker["def"] >= v[0] or attacker["atk"] + atk_bonus >= v[1]):
                extra += v[2]
            elif name == "sanct_higher" and self.der["players"][rel]["sanctuary"] > self.der["players"][1 - rel]["sanctuary"]:
                extra += v[0]
        code = attacker["code"]
        if code == PROFESSOR:      # +1 for each opponent field card (supports too) sharing an attribute with ours
            attrs = {self.db[c["code"]].attribute for c in mine if self.db.get(c["code"])}
            extra += sum(1 for f in self._field_cards(1 - rel) if self.db[f["code"]].attribute in attrs)
        elif code in ATTRIBUTE_ENABLERS:   # 오오미야 시노부 / 마츠바라 호노카: same rules as enables_kill
            rule = ATTRIBUTE_ENABLERS[code]
            if code == SHINOBU:
                extra += rule[1](self._field_attrs(1 - rel)) - 1
                if self._field_attrs(rel, skip=attacker) & rule[0]:
                    extra += 1
        if opt:
            free = [c for c in battle if c is not attacker and not (c["pos"] & POS_DEFENSE)]
            for name, g in t.dotte_options:
                v = [int(x) for x in g]
                if name == "opt_stiff" and len(free) >= v[0]:
                    extra += v[1]
                    break
                if name == "opt_stiff_each":
                    extra += v[0] * len(free)
                elif name == "opt_gauge" and self.der["players"][rel]["dotte"] >= t.dotte + v[0]:
                    extra += v[1]
        return extra

    def _dotte_damage_feats(self, o):
        """[damage, defeats, overkill, damage with the optional cost, defeats with it, HP removed] of the
        dotte effect this option uses (its own activation: the best target / all enemies; or the
        picked target of the activation in progress)."""
        zero = [0.0] * 6
        if o.card is None or o.noop or o.desc in NOOP_DESCS:
            return zero
        code, con, loc, seq = o.card
        rel = 0 if con == self.me else 1
        if o.kind in ACTIVATE_KINDS and rel == 0 and loc == LOCATION_MZONE and o.desc in DAMAGE_DESCS:
            attacker, picked = self.battle[0].get(seq), None
        elif o.kind == K_CARD_PICK and self.ctx and self.ctx[0] in ACTIVATE_KINDS and self.ctx[1] \
                and self.ctx[2] in DAMAGE_DESCS and self.ctx[1][2] == LOCATION_MZONE \
                and loc == LOCATION_MZONE and rel == 1:
            attacker, picked = self.battle[0].get(self.ctx[1][3]), self.battle[1].get(seq)
            if picked is None:
                return zero
        else:
            return zero
        if attacker is None:
            return zero
        t = text_of(self.db, attacker["code"])
        if not t.dotte_hits:
            return zero
        foes = list(self.battle[1].values())
        if not foes:
            return zero

        def hits(opt):
            return [(f, self.dotte_damage(t, attacker, f, opt)) for f in foes]

        base, full = hits(False), hits(True)
        has_opt = bool(t.dotte_options)
        if picked is not None or not t.dotte_to_all:
            # single target: this pick, or the best target of the activation
            def best(hs):
                if picked is not None:
                    return next(h for h in hs if h[0] is picked)
                return max(hs, key=lambda h: (h[1] >= derived.lethal_hp(h[0]), -abs(h[1] - derived.lethal_hp(h[0]))))
            f, dmg = best(base)
            kill = dmg >= derived.lethal_hp(f)
            f2, dmg2 = best(full)
            removed = sum(min(d, h["def"]) for h, d in base) if t.dotte_to_all else min(dmg, f["def"])
            return [dmg / 5.0, 1.0 if kill else 0.0, max(0, dmg - f["def"]) / 5.0 if kill else 0.0,
                    dmg2 / 5.0 if has_opt else 0.0,
                    1.0 if has_opt and dmg2 >= derived.lethal_hp(f2) else 0.0, removed / 10.0]
        kills = sum(1 for h, d in base if d >= derived.lethal_hp(h))
        kills2 = sum(1 for h, d in full if d >= derived.lethal_hp(h))
        dmg = max(d for _, d in base)
        return [dmg / 5.0, kills / 3.0, 0.0, max(d for _, d in full) / 5.0 if has_opt else 0.0,
                kills2 / 3.0 if has_opt else 0.0, sum(min(d, h["def"]) for h, d in base) / 10.0]

    def _effect_of(self, o):
        """(card code, desc, target) of what this option does: its own activation, or a target picked
        for the activation in progress (target = the picked battle creamate)."""
        if o.card is None or o.noop or o.desc in NOOP_DESCS:
            return None
        code, con, loc, seq = o.card
        rel = 0 if con == self.me else 1
        if o.kind in ACTIVATE_KINDS and rel == 0:
            return code, o.desc, None
        if o.kind == K_CARD_PICK and self.ctx and self.ctx[0] in ACTIVATE_KINDS and self.ctx[1] \
                and loc == LOCATION_MZONE:
            return self.ctx[1][0], self.ctx[2], self.battle[rel].get(seq)
        return None

    def _incoming_after(self, o):
        """Against the incoming attack: is the attacked creamate still defeated after this option's heal
        on it and / or sleep on the attacker (both count, e.g. 야마구치 키사라기: all allies +2 and sleep)?
        None when the option does neither."""
        eff = self._effect_of(o)
        if eff is None:
            return None
        code, desc, picked = eff
        attacker, target = self.incoming
        hp, atk, acts = target["def"], attacker["atk"], False
        amount, to_all, overheal = self._heal_of(code, desc)
        if amount and (picked is None or picked is target):
            hp += self._heal_on(target, amount, overheal)
            acts = True
        status = text_of(self.db, code).enemy_status
        if status and status[0] == COUNTER_SLEEP and desc not in COMMON_HEAL_DESCS \
                and (picked is None or picked is attacker):
            acts = True
            if attacker["counters"].get(COUNTER_SLEEP, 0) == 0:   # sleep does not stack
                atk = max(0, atk - 1)
        if not acts:
            return None
        return atk >= derived.lethal_hp(dict(target, **{"def": hp}))

    def _status_dup(self, o):
        """Share of the targets of this option's status effect that it would not change: they already have
        that status for at least as many turns (applying it again resets the turns left to the card's
        duration, so a shorter one left is lengthened)."""
        eff = self._effect_of(o)
        if eff is None:
            return 0.0
        code, desc, picked = eff
        status = text_of(self.db, code).enemy_status
        if not status or desc in COMMON_HEAL_DESCS:
            return 0.0   # the plain heal choice of a "① heal / ② status" card gives no status
        counter, _, turns = status
        if picked is not None:
            return 1.0 if picked["rel"] == 1 and picked["counters"].get(counter, 0) >= turns else 0.0
        foes = list(self.battle[1].values())
        if not foes:
            return 0.0
        return sum(1 for f in foes if f["counters"].get(counter, 0) >= turns) / len(foes)

    def _option_core(self, o):
        out = [0.0] * OPTION_EXTRA_DIM
        if o.card is None:
            return out
        code, con, loc, seq = o.card
        rel = 0 if con == self.me else 1
        allies = list(self.battle[0].values())
        if o.kind in ACTIVATE_KINDS and rel == 0:
            amount, to_all, overheal = self._heal_of(code, o.desc)
            if amount and allies:
                gains = [self._heal_on(c, amount, overheal) for c in allies]
                gain = sum(gains) if to_all else max(gains)
                out[0:3] = [1.0, gain / 5.0, 1.0 if gain == 0 else 0.0]
        elif o.kind == K_CARD_PICK and self.ctx and loc == LOCATION_MZONE:
            kind, ccard, cdesc = self.ctx
            target = self.battle[rel].get(seq)
            if target is None or not ccard:
                return out
            if kind in ACTIVATE_KINDS and rel == 0:
                amount, to_all, overheal = self._heal_of(ccard[0], cdesc)
                if amount and not to_all:
                    gain = self._heal_on(target, amount, overheal)
                    out[0:3] = [1.0, gain / 5.0, 1.0 if gain == 0 else 0.0]
            elif kind == K_BATTLE_ATTACK and rel == 1:
                attacker = self.battle[0].get(ccard[3]) if ccard[2] == LOCATION_MZONE else None
                if attacker is not None:
                    dmg, left, killed, over, back = self._hit(attacker["atk"], target)
                    out[3:8] = [dmg / 5.0, left / 10.0, 1.0 if killed else 0.0, over / 5.0, back / 5.0]
        elif o.kind == K_BATTLE_ATTACK and rel == 0 and loc == LOCATION_MZONE:
            attacker = self.battle[0].get(seq)
            foes = list(self.battle[1].values())
            if attacker is not None and foes:
                hits = [self._hit(attacker["atk"], f) for f in foes]
                survivors = [h for h in hits if not h[2]]
                out[3] = attacker["atk"] / 5.0
                out[5] = 1.0 if any(h[2] for h in hits) else 0.0
                out[7] = min(h[4] for h in survivors) / 5.0 if survivors else 0.0
        return out


SANCTUARY_HINT = (10050112 << 20) | 5   # Kirafan.drawop: choose 0-1 hand card to put under the summoner
# hand from which one card must go in: the opponent's turn (its end trims the hand to 5; a 드로우+2 card counts 2),
# our own turn (after its 3 draws; a 드로우+2 card counts 1)
SANCTUARY_RULE_HAND_OPP = 5
SANCTUARY_RULE_HAND_OWN = 8
SANCTUARY_RULE_LIMIT = 10                # 성채석 한도 (summoner defense) from which the model decides again
SANCTUARY_KEEP_BATTLE = 2                # battle creamates the hand keeps when the rule puts one in


def sanctuary_rule(view, decision, db):
    """Rule: in the draw phase's 성채석 choice, with a big hand (SANCTUARY_RULE_HAND_*) and a 성채석 한도 below
    SANCTUARY_RULE_LIMIT one hand card goes in - "pick nothing" is taken off the options, and so are the battle
    creamates while the hand holds no more than SANCTUARY_KEEP_BATTLE of them (unless nothing else is left); the
    model still picks which card, the one it needs least. Changes decision.options in place."""
    if decision.hint != SANCTUARY_HINT or decision.msg != MSG_SELECT_CARD or view.get("phase") != PHASE_DRAW:
        return
    if view["players"][0].get("summoner_def", 0) >= SANCTUARY_RULE_LIMIT:
        return
    own = view["turn_player"] == 0
    codes = [c["code"] for c in view["cards"] if c["rel"] == 0 and c["loc"] == LOCATION_HAND]
    hand = sum(2 if not own and text_of(db, c).draw >= 2 else 1 for c in codes)
    picks = [o for o in decision.options if o.kind == K_CARD_PICK]
    if hand < (SANCTUARY_RULE_HAND_OWN if own else SANCTUARY_RULE_HAND_OPP) or not picks:
        return

    def battle(o):
        card = db.get(o.card[0]) if o.card else None
        return card is not None and card.is_battle

    if sum(1 for c in codes if db.get(c) is not None and db.get(c).is_battle) <= SANCTUARY_KEEP_BATTLE:
        picks = [o for o in picks if not battle(o)] or picks
    decision.options = picks


def encode(view, decision, db, der=None):
    """Returns dict with g (global floats), cards/card_codes (token floats, vocab ids),
    opts (option floats) and opt_ids ([kind, card_vocab, desc_vocab, desc_idx, card_ptr]).
    Applies sanctuary_rule to decision.options first."""
    sanctuary_rule(view, decision, db)
    der = der or derived.compute(view, db)
    vocab = db.vocab            # effect descriptions / hints
    card_ids = db.card_vocab    # card identity (all summoners share one id)

    # ---- card tokens
    tokens = []
    for c in view["cards"]:
        if c["loc"] in TOKEN_LOCS:
            if c["loc"] == LOCATION_EXTRA and not (c["pos"] & POS_FACEUP):
                continue
            tokens.append(c)
    # keep field cards first if we need to truncate
    tokens.sort(key=lambda c: (c["loc"] not in (LOCATION_MZONE, LOCATION_SZONE), c["loc"] == LOCATION_GRAVE))
    tokens = tokens[:MAX_CARD_TOKENS]
    index = {}
    cards, codes = [], []
    for i, c in enumerate(tokens):
        index[(c["rel"], c["loc"], c["seq"])] = i
        dc = der["cards"].get(id(c))
        counters = c["counters"]
        f = [1.0 if c["rel"] == 0 else 0.0, 1.0 if c["rel"] == 1 else 0.0]
        f += _loc_onehot(c["loc"], c["seq"])
        f.append(c["seq"] / 7.0)
        f += [1.0 if c["pos"] & POS_FACEUP else 0.0, 1.0 if c["pos"] & POS_DEFENSE else 0.0]
        f += [c["atk"] / 10.0, c["def"] / 10.0, c["base_def"] / 10.0, c["level"] / 3.0, c["overlay"] / 10.0]
        f += _card_static(db, c["code"])
        f += [min(counters.get(k, 0), 5) / 5.0 for k in TRACKED_COUNTERS]
        if dc:
            f += [1.0 if dc["killable_single"] else 0.0, 1.0 if dc["killable_total"] else 0.0,
                  dc["lethal_hp"] / 10.0, 1.0 if dc["bound"] else 0.0]
        else:
            f += [0.0, 0.0, 0.0, 0.0]
        cards.append(f)
        codes.append(card_ids.get(c["code"], 0))

    # ---- global vector
    phase = view.get("phase", 0)
    g = [1.0 if phase == ph else 0.0 for ph in PHASES]
    g.append(1.0 if phase not in PHASES else 0.0)
    g += [1.0 if view["turn_player"] == 0 else 0.0, min(view.get("turn", 0), 40) / 20.0]
    g += [1.0 if decision.msg == m else 0.0 for m in MSG_TYPES]
    picked = len(decision.selection.picked) if decision.selection else 0
    g += [min(decision.min, 10) / 5.0, min(decision.max, 10) / 5.0, picked / 5.0]
    removed = ([0] * 5, [0] * 5)
    for c in view["cards"]:
        if c["loc"] == LOCATION_REMOVED and c["code"]:
            card = db.get(c["code"])
            if card is not None and card.class_index < 5:
                removed[c["rel"]][card.class_index] += 1
    for rel in (0, 1):
        g += _player_feats(view["players"][rel], der["players"][rel], removed[rel])
    g += [der["advantage"] / 10.0, 1.0 if der["realist"] else 0.0, 1.0 if der["mid"] else 0.0,
          1.0 if der["in_battle"] else 0.0, len(view.get("chain", ())) / 5.0]

    # ---- options
    hint_code = decision.hint >> 20
    tactics = _Tactics(view, decision, db, der)
    mine = list(tactics.battle[0].values())
    short = max((tactics.threat(c)[0] - derived.lethal_hp(c) for c in mine), default=0) if tactics.threat_turn else 0
    g += [tactics.slots_used[0] / 5.0, tactics.slots_used[1] / 5.0,
          sum(1 for c in mine if tactics.lethal_this_turn(c)) / 3.0, max(-10, min(10, short)) / 10.0,
          1.0 if tactics.can_heal[0] else 0.0, 1.0 if tactics.can_heal[1] else 0.0,
          tactics.opp_buff[0] / 3.0, tactics.opp_buff[1] / 3.0]
    extras = [tactics.option(o) for o in decision.options]
    useless = tactics.useless_heals(decision.options, extras)
    noop = [1 if (o.noop or u) else 0 for o, u in zip(decision.options, useless)]
    opts, opt_ids = [], []
    for o, extra, is_noop in zip(decision.options, extras, noop):
        f = [0.0, 0.0]
        card_vocab = 0
        ptr = 0
        if o.card is not None:
            code, con, loc, seq = o.card
            rel = 0 if con == view["me"] else 1
            f = [1.0 if rel == 0 else 0.0, 1.0 if rel == 1 else 0.0]
            f += _loc_onehot(loc & 0x7f, seq)
            f.append(min(seq, 7) / 7.0)
            card_vocab = card_ids.get(code, 0)
            ptr = index.get((rel, loc, seq), -1) + 1
        else:
            f += [0.0] * len(CARD_LOCS) + [0.0]
        desc = o.desc if o.kind not in (0, 1, 2, 3, 4) else 0
        dcode, didx = desc >> 20, desc & 0xfffff
        if o.kind == 23:  # announce number: desc is the number itself
            f += [min(desc, 20) / 10.0, 0.0, float(o.extra), 1.0 if hint_code else 0.0]
            dcode, didx = 0, 0
        else:
            f += [0.0, 1.0 if dcode == 0 and desc else 0.0, float(o.extra), 1.0 if hint_code else 0.0]
        f.append(float(is_noop))
        f += extra
        opts.append(f)
        opt_ids.append([o.kind, card_vocab, vocab.get(dcode, 0), min(didx, 31) if dcode else 0, ptr])
    # The policy adds a large negative logit bias to no-op options unless all options are no-ops
    if all(noop):
        noop = [0] * len(noop)

    return {"g": g, "cards": cards, "card_codes": codes, "opts": opts, "opt_ids": opt_ids, "noop": noop,
            "hint": [vocab.get(hint_code, 0), (decision.hint & 0xfffff) if hint_code else 0]}


def pack(enc):
    """Compact binary form of encode() output (float32/int32 buffers), cheap to pickle and to
    turn into numpy arrays with np.frombuffer on the learner side."""
    flat_cards = [x for row in enc["cards"] for x in row]
    flat_opts = [x for row in enc["opts"] for x in row]
    flat_ids = [x for row in enc["opt_ids"] for x in row]
    return {
        "packed": 1,
        "g": array("f", enc["g"]).tobytes(),
        "cards": array("f", flat_cards).tobytes(),
        "card_codes": array("i", enc["card_codes"]).tobytes(),
        "opts": array("f", flat_opts).tobytes(),
        "opt_ids": array("i", flat_ids).tobytes(),
        "noop": array("f", enc["noop"]).tobytes(),
        "hint": array("i", enc["hint"]).tobytes(),
    }


assert NUM_KINDS > 0
