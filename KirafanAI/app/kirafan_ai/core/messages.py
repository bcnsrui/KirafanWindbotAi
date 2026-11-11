"""Parsing of ocgcore decision messages into option lists, and encoding of responses.

A decision is turned into a flat list of options. Multi-card selections are split into
one-card-at-a-time steps (see SelectionState) so the policy always picks exactly one option.
Decisions whose outcome does not matter for this game (zones, positions, sorting) are
answered automatically by `auto_response`.
"""
import itertools
import random
import struct

from .constants import *  # noqa: F401,F403
from .reader import Reader

# Option kinds (stable ids used by the model)
K_IDLE_SUMMON = 0
K_IDLE_SPSUMMON = 1
K_IDLE_REPOS = 2
K_IDLE_MSET = 3
K_IDLE_SSET = 4
K_IDLE_ACTIVATE = 5
K_IDLE_TO_BP = 6
K_IDLE_TO_EP = 7
K_IDLE_SHUFFLE = 8
K_BATTLE_ACTIVATE = 9
K_BATTLE_ATTACK = 10
K_BATTLE_TO_M2 = 11
K_BATTLE_TO_EP = 12
K_EFFECTYN_YES = 13
K_EFFECTYN_NO = 14
K_YESNO_YES = 15
K_YESNO_NO = 16
K_OPTION = 17
K_CARD_PICK = 18
K_CARD_FINISH = 19
K_CARD_CANCEL = 20
K_CHAIN_ACTIVATE = 21
K_CHAIN_PASS = 22
K_NUMBER = 23
K_UNSELECT_PICK = 24
K_UNSELECT_DROP = 25
K_UNSELECT_FINISH = 26
K_SUM_SUBSET = 27
NUM_KINDS = 28

KIND_NAMES = {
    K_IDLE_SUMMON: "일반소환", K_IDLE_SPSUMMON: "소환", K_IDLE_REPOS: "표시형식", K_IDLE_MSET: "세트",
    K_IDLE_SSET: "마함세트", K_IDLE_ACTIVATE: "효과발동", K_IDLE_TO_BP: "배틀페이즈로", K_IDLE_TO_EP: "엔드페이즈로",
    K_IDLE_SHUFFLE: "패셔플", K_BATTLE_ACTIVATE: "효과발동", K_BATTLE_ATTACK: "통상공격", K_BATTLE_TO_M2: "메인2로",
    K_BATTLE_TO_EP: "엔드페이즈로", K_EFFECTYN_YES: "효과사용:예", K_EFFECTYN_NO: "효과사용:아니오",
    K_YESNO_YES: "예", K_YESNO_NO: "아니오", K_OPTION: "옵션", K_CARD_PICK: "카드선택", K_CARD_FINISH: "선택완료",
    K_CARD_CANCEL: "취소", K_CHAIN_ACTIVATE: "체인발동", K_CHAIN_PASS: "체인안함", K_NUMBER: "숫자선언",
    K_UNSELECT_PICK: "카드선택", K_UNSELECT_DROP: "선택해제", K_UNSELECT_FINISH: "선택완료", K_SUM_SUBSET: "합계선택",
}

POLICY_MESSAGES = {
    MSG_SELECT_BATTLECMD, MSG_SELECT_IDLECMD, MSG_SELECT_EFFECTYN, MSG_SELECT_YESNO,
    MSG_SELECT_OPTION, MSG_SELECT_CARD, MSG_SELECT_CHAIN, MSG_SELECT_UNSELECT_CARD,
    MSG_ANNOUNCE_NUMBER, MSG_SELECT_TRIBUTE, MSG_SELECT_SUM,
}


def _sid(code, idx):
    return (code << 20) | idx


# Options that exist so a human can undo a misclick; choosing them leaves the game unchanged.
NOOP_DESCS = {
    _sid(10050111, 1),   # 클래스 별로 패 정렬하기
    _sid(10050111, 6),   # 돗테오키 효과 발동X
    _sid(10050111, 8),   # 효과 발동X
}
ACTIVATE_KINDS = {K_IDLE_ACTIVATE, K_BATTLE_ACTIVATE, K_CHAIN_ACTIVATE}
# Choices whose follow-up questions (targets, ...) belong to them: Decision.context
CONTEXT_KINDS = ACTIVATE_KINDS | {K_BATTLE_ATTACK, K_IDLE_SUMMON, K_IDLE_SPSUMMON}
# Options that always move the game forward; never treated as no-ops
PROGRESS_KINDS = {K_IDLE_TO_BP, K_IDLE_TO_EP, K_BATTLE_TO_M2, K_BATTLE_TO_EP, K_CHAIN_PASS, K_EFFECTYN_NO,
                  K_YESNO_NO, K_CARD_FINISH, K_UNSELECT_FINISH}


class Option:
    __slots__ = ("kind", "card", "desc", "value", "extra", "noop")

    def __init__(self, kind, card=None, desc=0, value=0, extra=0, noop=False):
        self.kind = kind
        self.card = card      # (code, controller, location, sequence) or None
        self.desc = desc      # effect description (aux.Stringid) or option value
        self.value = value    # response payload component
        self.extra = extra    # kind-specific (e.g. direct attack flag)
        self.noop = noop      # leaves the game state unchanged: always the lowest priority

    def __repr__(self):
        return "Option(%s,%r,%x,%r%s)" % (KIND_NAMES.get(self.kind, self.kind), self.card, self.desc, self.value,
                                          ",noop" if self.noop else "")


def mark_static_noops(options):
    for o in options:
        if (o.kind in ACTIVATE_KINDS and o.desc in NOOP_DESCS) or o.kind == K_UNSELECT_DROP:
            o.noop = True


class Decision:
    """A pending choice for `player`. `options` is what the policy picks from."""

    def __init__(self, msg, player, raw):
        self.msg = msg
        self.player = player
        self.raw = raw
        self.options = []
        self.cards = []          # candidate cards for card-type selections
        self.min = 0
        self.max = 0
        self.cancelable = False
        self.selection = None    # SelectionState for multi-card picks
        self.hint = 0            # last HINT_SELECTMSG for this player
        self.meta = {}
        self.context = None      # (kind, card, desc) of this player's last activation / attack / summon

    def __repr__(self):
        return "Decision(msg=%d, player=%d, %d options)" % (self.msg, self.player, len(self.options))


def _card(r, long_seq=True):
    code = r.u32()
    con = r.u8()
    loc = r.u8()
    seq = r.u32() if long_seq else r.u8()
    return code, con, loc, seq


def _card_locinfo(r):
    code = r.u32()
    con, loc, seq, pos = r.loc_info()
    return code, con, loc, seq


def parse_decision(raw, hint=0):
    """Return a Decision for a decision message, or None if it is not a decision."""
    r = Reader(raw)
    msg = r.u8()
    if msg not in DECISION_MESSAGES:
        return None
    player = r.u8()
    d = Decision(msg, player, raw)
    d.hint = hint
    o = d.options

    if msg == MSG_SELECT_IDLECMD:
        for kind in (K_IDLE_SUMMON, K_IDLE_SPSUMMON):
            for i in range(r.u32()):
                o.append(Option(kind, _card(r), value=(i << 16) | (kind - K_IDLE_SUMMON)))
        for i in range(r.u32()):
            o.append(Option(K_IDLE_REPOS, _card(r, long_seq=False), value=(i << 16) | 2))
        for kind, t in ((K_IDLE_MSET, 3), (K_IDLE_SSET, 4)):
            for i in range(r.u32()):
                o.append(Option(kind, _card(r), value=(i << 16) | t))
        for i in range(r.u32()):
            c = _card(r)
            desc = r.u64()
            r.u8()
            o.append(Option(K_IDLE_ACTIVATE, c, desc=desc, value=(i << 16) | 5))
        if r.u8():
            o.append(Option(K_IDLE_TO_BP, value=6))
        if r.u8():
            o.append(Option(K_IDLE_TO_EP, value=7))
        r.u8()  # shuffle hand: never useful, not offered

    elif msg == MSG_SELECT_BATTLECMD:
        for i in range(r.u32()):
            c = _card(r)
            desc = r.u64()
            r.u8()
            o.append(Option(K_BATTLE_ACTIVATE, c, desc=desc, value=(i << 16) | 0))
        for i in range(r.u32()):
            c = _card(r, long_seq=False)
            direct = r.u8()
            o.append(Option(K_BATTLE_ATTACK, c, value=(i << 16) | 1, extra=direct))
        if r.u8():
            o.append(Option(K_BATTLE_TO_M2, value=2))
        if r.u8():
            o.append(Option(K_BATTLE_TO_EP, value=3))

    elif msg == MSG_SELECT_EFFECTYN:
        c = _card_locinfo(r)
        desc = r.u64()
        d.meta["card"] = c
        o.append(Option(K_EFFECTYN_YES, c, desc=desc, value=1))
        o.append(Option(K_EFFECTYN_NO, c, desc=desc, value=0))

    elif msg == MSG_SELECT_YESNO:
        desc = r.u64()
        o.append(Option(K_YESNO_YES, desc=desc, value=1))
        o.append(Option(K_YESNO_NO, desc=desc, value=0))

    elif msg == MSG_SELECT_OPTION:
        for i in range(r.u8()):
            o.append(Option(K_OPTION, desc=r.u64(), value=i))

    elif msg in (MSG_SELECT_CARD, MSG_SELECT_TRIBUTE):
        d.cancelable = bool(r.u8())
        d.min = r.u32()
        d.max = r.u32()
        for i in range(r.u32()):
            if msg == MSG_SELECT_CARD:
                d.cards.append(_card_locinfo(r))
            else:
                d.cards.append(_card(r))
                r.u8()
        d.selection = SelectionState(d)
        d.selection.refresh_options()

    elif msg == MSG_SELECT_UNSELECT_CARD:
        finishable = r.u8()
        cancelable = r.u8()
        d.min = r.u32()
        d.max = r.u32()
        sel = [_card_locinfo(r) for _ in range(r.u32())]
        unsel = [_card_locinfo(r) for _ in range(r.u32())]
        d.meta["n_select"] = len(sel)
        for i, c in enumerate(sel):
            o.append(Option(K_UNSELECT_PICK, c, value=i))
        for i, c in enumerate(unsel):
            o.append(Option(K_UNSELECT_DROP, c, value=len(sel) + i))
        if finishable or cancelable:
            o.append(Option(K_UNSELECT_FINISH, value=-1))

    elif msg == MSG_SELECT_CHAIN:
        r.u8()  # spe_count
        forced = r.u8()
        r.u32()
        r.u32()
        for i in range(r.u32()):
            c = _card_locinfo(r)
            desc = r.u64()
            r.u8()
            o.append(Option(K_CHAIN_ACTIVATE, c, desc=desc, value=i))
        if not forced:
            o.append(Option(K_CHAIN_PASS, value=-1))
        d.meta["forced"] = forced

    elif msg == MSG_ANNOUNCE_NUMBER:
        for i in range(r.u8()):
            o.append(Option(K_NUMBER, desc=r.u64(), value=i))

    elif msg == MSG_SELECT_SUM:
        mode_any = r.u8()  # 0: exact sum with count in [min,max]; 1: sum >= acc
        acc = r.u32()
        d.min = r.u32()
        d.max = r.u32()
        must = []
        for _ in range(r.u32()):
            c = _card_locinfo(r)
            must.append((c, r.u32()))
        cand = []
        for _ in range(r.u32()):
            c = _card_locinfo(r)
            cand.append((c, r.u32()))
        d.cards = [c for c, _ in cand]
        for subset in _sum_subsets(must, cand, acc, d.min, d.max, mode_any):
            o.append(Option(K_SUM_SUBSET, d.cards[subset[0]] if subset else None, value=subset))

    else:
        d.meta["auto"] = True
        _parse_auto_meta(d, r)
    mark_static_noops(d.options)
    return d


def _parse_auto_meta(d, r):
    msg = d.msg
    if msg in (MSG_SELECT_PLACE, MSG_SELECT_DISFIELD):
        d.meta["count"] = r.u8()
        d.meta["flag"] = r.u32()
    elif msg == MSG_SELECT_POSITION:
        r.u32()
        d.meta["positions"] = r.u8()
    elif msg in (MSG_SORT_CARD, MSG_SORT_CHAIN):
        pass
    elif msg == MSG_SELECT_COUNTER:
        r.u16()
        count = r.u16()
        avail = []
        for _ in range(r.u32()):
            r.u32(); r.u8(); r.u8(); r.u8()
            avail.append(r.u16())
        d.meta["count"] = count
        d.meta["avail"] = avail
    elif msg == MSG_ANNOUNCE_RACE:
        d.meta["count"] = r.u8()
        d.meta["available"] = r.u64()
    elif msg == MSG_ANNOUNCE_ATTRIB:
        d.meta["count"] = r.u8()
        d.meta["available"] = r.u32()
    elif msg == MSG_ANNOUNCE_CARD:
        d.meta["opcodes"] = [r.u64() for _ in range(r.u8())]


def _sum_ok(params, acc, mode_any):
    if mode_any:
        lo = sum(min(p & 0xffff, (p >> 16) or (p & 0xffff)) for p in params)
        hi = sum(max(p & 0xffff, p >> 16) for p in params)
        mn = min((min(p & 0xffff, (p >> 16) or (p & 0xffff)) for p in params), default=0)
        return hi >= acc and lo - mn < acc

    def rec(i, rest):
        if i == len(params):
            return rest == 0
        o1, o2 = params[i] & 0xffff, params[i] >> 16
        return (rest >= o1 and rec(i + 1, rest - o1)) or (o2 > 0 and rest >= o2 and rec(i + 1, rest - o2))
    return rec(0, acc)


def _sum_subsets(must, cand, acc, mn, mx, mode_any, limit=48):
    must_params = [p for _, p in must]
    out = []
    n = len(cand)
    hi = n if mode_any else min(n, max(mx, mn))
    for k in range(0 if mode_any else max(mn, 0), hi + 1):
        for comb in itertools.combinations(range(n), k):
            if _sum_ok(must_params + [cand[i][1] for i in comb], acc, mode_any):
                out.append(list(comb))
                if len(out) >= limit:
                    return out
    return out


class SelectionState:
    """Splits MSG_SELECT_CARD/TRIBUTE (pick min..max cards) into single picks."""

    def __init__(self, decision):
        self.d = decision
        self.picked = []

    def refresh_options(self):
        d = self.d
        d.options = []
        for i, c in enumerate(d.cards):
            if i not in self.picked:
                d.options.append(Option(K_CARD_PICK, c, value=i))
        n = len(self.picked)
        if n >= d.min and n > 0:
            d.options.append(Option(K_CARD_FINISH, value=-2))
        if n == 0 and (d.cancelable or d.min == 0):
            # With min == 0 this means "pick nothing", a real choice; otherwise it undoes the activation
            d.options.append(Option(K_CARD_CANCEL, value=-1, noop=d.min > 0))

    def apply(self, opt):
        """Returns the response bytes when the selection is complete, else None."""
        d = self.d
        if opt.kind == K_CARD_CANCEL:
            return struct.pack("<i", -1)
        if opt.kind == K_CARD_PICK:
            self.picked.append(opt.value)
        if opt.kind == K_CARD_FINISH or len(self.picked) >= min(d.max, len(d.cards)):
            return encode_card_list(self.picked)
        self.refresh_options()
        return None


def encode_card_list(indices):
    return struct.pack("<iI%dI" % len(indices), 0, len(indices), *indices)


def encode_response(d, opt):
    """Response bytes for a single-step decision."""
    msg = d.msg
    if msg in (MSG_SELECT_IDLECMD, MSG_SELECT_BATTLECMD, MSG_SELECT_EFFECTYN, MSG_SELECT_YESNO,
               MSG_SELECT_OPTION, MSG_SELECT_CHAIN, MSG_ANNOUNCE_NUMBER):
        return struct.pack("<i", opt.value)
    if msg == MSG_SELECT_UNSELECT_CARD:
        if opt.kind == K_UNSELECT_FINISH:
            return struct.pack("<i", -1)
        return struct.pack("<ii", 1, opt.value)
    if msg == MSG_SELECT_SUM:
        return encode_card_list(opt.value)
    raise ValueError("unsupported decision %d" % msg)


MZONE_ORDER = (5, 6, 2, 1, 3, 0, 4)
SZONE_ORDER = (2, 1, 3, 0, 4, 5, 6, 7)


def auto_response(d, rng=random):
    """Responses for decisions that the policy does not control."""
    msg = d.msg
    m = d.meta
    if msg in (MSG_SELECT_PLACE, MSG_SELECT_DISFIELD):
        flag = m["flag"]
        out = b""
        for _ in range(m["count"]):
            choice = None
            # Summoners must land in the extra monster zone, so prefer it, then the center column
            for base, loc, order, owner in ((0, LOCATION_MZONE, MZONE_ORDER, 0), (8, LOCATION_SZONE, SZONE_ORDER, 0),
                                            (16, LOCATION_MZONE, MZONE_ORDER, 1),
                                            (24, LOCATION_SZONE, SZONE_ORDER, 1)):
                for s in order:
                    bit = 1 << (base + s)
                    if not (flag & bit):
                        choice = (owner, loc, s, bit)
                        break
                if choice:
                    break
            if choice is None:
                choice = (0, LOCATION_MZONE, 0, 0)
            owner, loc, s, bit = choice
            flag |= bit
            player = d.player if owner == 0 else 1 - d.player
            out += struct.pack("<BBB", player, loc, s)
        return out
    if msg == MSG_SELECT_POSITION:
        pos = m["positions"]
        for p in (POS_FACEUP_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE, POS_FACEDOWN_ATTACK):
            if pos & p:
                return struct.pack("<i", p)
        return struct.pack("<i", POS_FACEUP_ATTACK)
    if msg in (MSG_SORT_CARD, MSG_SORT_CHAIN):
        return struct.pack("<b", -1)
    if msg == MSG_SELECT_COUNTER:
        need = m["count"]
        out = []
        for a in m["avail"]:
            take = min(a, need)
            out.append(take)
            need -= take
        return struct.pack("<%dh" % len(out), *out)
    if msg == MSG_ANNOUNCE_RACE:
        avail, cnt, val = m["available"], m["count"], 0
        for b in range(64):
            if cnt and avail & (1 << b):
                val |= 1 << b
                cnt -= 1
        return struct.pack("<Q", val)
    if msg == MSG_ANNOUNCE_ATTRIB:
        avail, cnt, val = m["available"], m["count"], 0
        for b in range(32):
            if cnt and avail & (1 << b):
                val |= 1 << b
                cnt -= 1
        return struct.pack("<I", val)
    if msg == MSG_ANNOUNCE_CARD:
        return struct.pack("<i", 0)
    if msg == MSG_ROCK_PAPER_SCISSORS:
        return struct.pack("<i", rng.randint(1, 3))
    raise ValueError("no auto response for %d" % msg)
