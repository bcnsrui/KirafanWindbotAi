"""Rough numbers read from card texts (expansions/texts.csv via the cdb), for the rule-based AI."""
import re

_DOTTE = re.compile(r"\[돗테오키 게이지\s*(\d+)")
_EXTRA_STIFF = re.compile(r"추가 경직\s*(\d+)")
_SINGLE_DMG = re.compile(r"적 단일에게\s*(\d+)\s*데미지")
_ALL_DMG = re.compile(r"적 전체에게\s*(\d+)\s*데미지")
_HEAL_SINGLE = re.compile(r"아군 단일을\s*(\d+)\s*회복")
_HEAL_ALL = re.compile(r"아군 전체를\s*(\d+)\s*회복")
_DRAW = re.compile(r"드로우\+(\d+)")
_ATK_UP = re.compile(r"공격력\+(\d+)")
_OVERHEAL = "최대 체력을 초과해 회복"
# "적 전체에게 다음 턴까지 수면을 부여한다." -> (counter, to all enemies?)
_ENEMY_STATUS = re.compile(r"적 (단일|전체)에게[^/\n]*?(나약함|불행|속박|배고픔|수면|고립)을? 부여")
STATUS_COUNTERS = {"나약함": 0xb01, "불행": 0xb02, "속박": 0xb03, "배고픔": 0xb04, "수면": 0xb05, "고립": 0xb06}
# how long a status lasts (Kirafan6.statuseff sets the counter to this; every end phase takes 1 off):
# "이번 턴" 1, "다음 턴까지" 2, "N턴간" N
_STATUS_TURNS = ((re.compile(r"(\d+)턴간"), None), (re.compile(r"다음 턴까지"), 2), (re.compile(r"이번 턴"), 1))


_DOTTE_SECTION = re.compile(r"\[돗테오키 게이지[^\]]*\](.*?)(?=\[|$)", re.S)
# conditional extra damage of dotte effects: (rule name, regex); the groups are the condition's
# parameters followed by the bonus
_DMG_RULES = (
    ("my_battle_ge", re.compile(r"자신 필드의 배틀 크리에메이트가\s*(\d+)장 이상이면 추가 데미지\+(\d+)")),
    ("my_field_ge", re.compile(r"자신 필드의 크리에메이트가\s*(\d+)장 이상이면 추가 데미지\+(\d+)")),
    ("target_status", re.compile(r"(나약함|불행|속박|배고픔|수면|고립) 상태의 크리에메이트에게는 추가 데미지\+(\d+)")),
    ("target_class", re.compile(r"(전사|마법사|나이트) 크리에메이트에게는 추가 데미지\+(\d+)")),
    ("target_attr", re.compile(r"(땅|물|불|바람|해|달) 속성 크리에메이트에게는 추가 데미지\+(\d+)")),
    ("opp_hand_ge", re.compile(r"상대 패가\s*(\d+)장 이상이면 추가 데미지\+(\d+)")),
    ("no_mage", re.compile(r"자신 필드에 마법사 크리에메이트가 없으면 추가 데미지\+(\d+)")),
    ("no_support", re.compile(r"자신 서포트 존에 아군이 없으면 추가 데미지\+(\d+)")),
    ("other_attr", re.compile(r"자신 필드에 이 카드 이외의 (땅|물|불|바람|해|달) 속성 크리에메이트가 있으면 추가 데미지\+(\d+)")),
    ("self_hp_le_minus", re.compile(r"자신 체력이\s*(\d+) 이하면 추가 데미지-(\d+)")),
    ("self_hp_ge", re.compile(r"자신 체력이\s*(\d+) 이상이면 추가 데미지\+(\d+)")),
    ("self_hp_or_atk", re.compile(r"자신 체력이\s*(\d+) 이상이거나 공격력\s*(\d+) 이상이면 추가 데미지\+(\d+)")),
    ("sanct_higher", re.compile(r"성전에 있는 크리에메이트의 레벨 합계가 상대보다 높으면 추가 데미지\+(\d+)")),
)
# extra damage for an optional extra cost
_DMG_OPTIONS = (
    ("opt_stiff", re.compile(r"추가로\s*(\d+)장 경직하면 추가 데미지\+(\d+)")),
    ("opt_stiff_each", re.compile(r"추가로 경직한 수만큼 추가 데미지\+(\d+)")),
    ("opt_gauge", re.compile(r"돗테오키 게이지\s*(\d+)를 사용하여 추가 데미지\+(\d+)")),
)
# dotte damage that depends on the field: "자신 필드의 땅 속성 크리에메이트 수+1" (미니와 츠미키), "상대 배틀 크리에메이트
# 수만큼" (우시쿠 카나)
_VAR_ATTR_DMG = re.compile(r"적 (단일|전체)에게 자신 필드의 (땅|물|불|바람|해|달) 속성 크리에메이트 수\+(\d+) 데미지")
_VAR_OPP_DMG = re.compile(r"적 (단일|전체)에게 상대 배틀 크리에메이트 수만큼 데미지")
# Kirafan2.Allbattleop (set code 0xa02): every enemy takes floor(attack / 2)
_HALF_ALL = "자신의 통상 공격은 적 전체에게 절반의 데미지"
_UNSTIFF_AFTER_ATTACK = "경직 상태의 자신은 통상 공격 후에 경직 상태가 해제된다"   # 혼다 타마키
_ATTACK_GAUGE = re.compile(r"자신의 통상 공격시 증가하는 돗테오키 게이지는\s*(\d+)가 된다")   # 소냐
_ATTACK_WEAK = "자신의 통상 공격은 추가로 이번 턴 나약함을 부여"   # 시마 린
_NOT_ATTACKABLE = "자신 배틀 존에 다른 아군이 있으면 상대는 자신을 통상 공격할 수 없다"   # 봇치
_VS_ATTR_ATK = re.compile(r"(땅|물|불|바람|해|달) 속성 크리에메이트에게 통상 공격시 공격력\+(\d+)")
_SELF_ATK = re.compile(r"이번 턴 자신의 공격력\+(\d+)")
_SELF_ATK_OPP = "상대 배틀 크리에메이트 수만큼 자신의 공격력 증가"   # 소노다 유우 (3 turns)
# more normal attacks: (extra stiff it needs, whose: 자신 / 아군 단일 / 아군 전체, +N)
_MORE_ATTACKS = re.compile(r"(?:자신 필드의 배틀 크리에메이트를 추가로\s*(\d+)장 경직하면\s*)?(자신|아군 단일|아군 전체)의 "
                           r"통상 공격 횟수\+(\d+)")
_WHO = {"자신": "self", "아군 단일": "one", "아군 전체": "all"}
_GAUGE_PLUS = re.compile(r"돗테오키 게이지\+(\d+)")
_GAUGE_OPP_FIELD = re.compile(r"상대 필드의 크리에메이트 수\+(\d+)만큼 돗테오키 게이지 증가")   # 텐노지 나기사
_STIFF_HEAD = re.compile(r"추가 경직\s*(\d+)장")
# "[자신 배틀 페이즈 개시시]" effects on attack
_START_ALLIES = re.compile(r"아군 전체\(([^)]*)\)의 (다음 공격이 1번만 )?공격력\+(\d+)")
_START_SELF_COND = re.compile(r"자신 체력이\s*(\d+) 이상이거나 공격력\s*(\d+) 이상이면 이번 턴 자신의 공격력\+(\d+)")
_START_HAND_DIFF = "자신 패가 상대보다 적으면 그 차이만큼"          # 히바리가오카 루리
_START_HUNGER = "배고픔 상태의 크리에메이트의 수만큼 자신의 공격력"   # 히라사와 유이
# "[패시브]" attack (and level) that follows the board: (rule, regex); the groups are its parameters
_PASSIVE_ATK = (
    ("field_ge", re.compile(r"자신 필드의 크리에메이트가\s*(\d+)장 이상이면 공격력\+(\d+)")),
    ("attr_ge", re.compile(r"자신 필드의 (땅|물|불|바람|해|달) 속성 크리에메이트가\s*(\d+)장 이상이면 공격력\+(\d+)")),
    ("hp_le", re.compile(r"자신 체력이\s*(\d+) 이하면 공격력-(\d+)")),
    ("hp_ge", re.compile(r"자신 체력이\s*(\d+) 이상이면 공격력\+(\d+)")),
    ("no_support", re.compile(r"자신 서포트 존에 아군이 없으면 자신의 공격력\+(\d+)")),
    ("sanct_higher", re.compile(r"자신 성전에 있는 크리에메이트의 레벨 합계가 상대보다 높으면 공격력\+(\d+)")),
    ("professor", re.compile(r"자신 필드에 존재하는 속성과 같은 상대 필드의 크리에메이트 수만큼 자신의 공격력 증가")),
    ("with_card", re.compile(r'자신 필드에 "([^"]+)"이 있으면 패 / 필드에 있는 자신의 레벨-(\d+) 공격력\+(\d+)')),
)
_SECTION = re.compile(r"\[([^\]]*)\]")


def sections(desc):
    """[(header, body)] of a card text: "[스킬] ...", "[돗테오키 게이지 2, 추가 경직 1장] ..." and so on."""
    parts = _SECTION.split(desc)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


class CardText:
    __slots__ = ("dotte", "extra_stiff", "single_dmg", "all_dmg", "heal_single", "heal_all", "draw", "atk_up",
                 "overheal", "enemy_status", "dotte_heal_single", "dotte_heal_all", "dotte_single", "dotte_all",
                 "dotte_rules", "dotte_options", "dotte_var", "dotte_hits", "dotte_to_all", "dotte_stiff",
                 "dotte_weak", "dotte_self_atk", "dotte_self_atk_opp", "dotte_more_attacks", "half_all",
                 "unstiff_after_attack", "not_attackable", "vs_attr_atk", "start", "gauge_gain", "more_attacks",
                 "attack_gauge", "attack_weak", "passive_atk")

    def __init__(self, desc):
        def first(rx):
            m = rx.search(desc)
            return int(m.group(1)) if m else 0
        self.dotte = first(_DOTTE)
        self.extra_stiff = first(_EXTRA_STIFF)
        self.single_dmg = first(_SINGLE_DMG)
        self.all_dmg = first(_ALL_DMG)
        self.heal_single = first(_HEAL_SINGLE)
        self.heal_all = first(_HEAL_ALL)
        self.draw = sum(int(x) for x in _DRAW.findall(desc))
        self.atk_up = first(_ATK_UP)
        self.overheal = _OVERHEAL in desc
        m = _ENEMY_STATUS.search(desc)
        # (counter, to all enemies, turns it lasts)
        self.enemy_status = (STATUS_COUNTERS[m.group(2)], m.group(1) == "전체", status_turns(desc, m)) if m else None
        # heals that are part of the dotte effect (e.g. 야마구치 키사라기: all allies +2 and sleep)
        sec = _DOTTE_SECTION.search(desc)
        sec = sec.group(1) if sec else ""
        m = _HEAL_SINGLE.search(sec)
        self.dotte_heal_single = int(m.group(1)) if m else 0
        m = _HEAL_ALL.search(sec)
        self.dotte_heal_all = int(m.group(1)) if m else 0
        # damage of the dotte effect: base single / all, conditional bonuses, optional-cost bonuses
        m = _SINGLE_DMG.search(sec)
        self.dotte_single = int(m.group(1)) if m else 0
        m = _ALL_DMG.search(sec)
        self.dotte_all = int(m.group(1)) if m else 0
        if "선택 효과" in sec and self.dotte_single:
            self.dotte_all = 0   # "① single / ② all" (카후우 치노): counted as the single choice
        self.dotte_rules = [(name, m.groups()) for name, rx in _DMG_RULES for m in rx.finditer(sec)]
        self.dotte_options = [(name, m.groups()) for name, rx in _DMG_OPTIONS for m in rx.finditer(sec)]
        # (to all enemies, "attr", attribute, +N) / (to all enemies, "opp_battle", None, 0)
        m = _VAR_ATTR_DMG.search(sec)
        m2 = _VAR_OPP_DMG.search(sec)
        self.dotte_var = (m.group(1) == "전체", "attr", m.group(2), int(m.group(3))) if m else \
            ((m2.group(1) == "전체", "opp_battle", None, 0) if m2 else None)
        self.dotte_hits = bool(self.dotte) and bool(self.dotte_single or self.dotte_all or self.dotte_var)
        self.dotte_to_all = self.dotte_hits and not self.dotte_single and \
            (bool(self.dotte_all) or bool(self.dotte_var and self.dotte_var[0]))
        head = re.search(r"\[(돗테오키 게이지[^\]]*)\]", desc)
        m = _STIFF_HEAD.search(head.group(1)) if head else None
        self.dotte_stiff = int(m.group(1)) if m else 0
        m = _ENEMY_STATUS.search(sec)
        self.dotte_weak = ("all" if m.group(1) == "전체" else "one") if m and m.group(2) == "나약함" else None
        m = _SELF_ATK.search(sec)
        self.dotte_self_atk = int(m.group(1)) if m else 0
        self.dotte_self_atk_opp = _SELF_ATK_OPP in sec
        m = _MORE_ATTACKS.search(sec)
        self.dotte_more_attacks = (int(m.group(1) or 0), _WHO[m.group(2)], int(m.group(3))) if m else None
        self.half_all = _HALF_ALL in desc
        self.unstiff_after_attack = _UNSTIFF_AFTER_ATTACK in desc
        m = _ATTACK_GAUGE.search(desc)
        self.attack_gauge = int(m.group(1)) if m else 1   # gauge its first normal attack of the turn adds
        self.attack_weak = _ATTACK_WEAK in desc
        self.not_attackable = _NOT_ATTACKABLE in desc
        m = _VS_ATTR_ATK.search(desc)
        self.vs_attr_atk = (m.group(1), int(m.group(2))) if m else None
        self.start = []
        self.passive_atk = []     # [(rule, parameters...)] (_PASSIVE_ATK)
        self.gauge_gain = []      # supports: [(gauge gain or -N: opponent's field creamates + N, gauge cost, stiff)]
        self.more_attacks = []    # supports: [(whose: "one" / "all", +N, gauge cost, stiff)]
        for head, body in sections(desc):
            if head.startswith("패시브"):
                self.passive_atk += [(name,) + m.groups() for name, rx in _PASSIVE_ATK for m in rx.finditer(body)]
                continue
            if head.startswith("자신 배틀 페이즈 개시시"):
                for m in _START_ALLIES.finditer(body):
                    self.start.append(("allies", tuple(m.group(1).split("+")), bool(m.group(2)), int(m.group(3))))
                m = _START_SELF_COND.search(body)
                if m:
                    self.start.append(("self_cond", int(m.group(1)), int(m.group(2)), int(m.group(3))))
                if _START_HAND_DIFF in body:
                    self.start.append(("self_hand_diff",))
                if _START_HUNGER in body:
                    self.start.append(("self_hunger",))
                continue
            usable = head.startswith("스킬") or head.startswith("돗테오키 게이지") or head.startswith("등장")
            if not usable:
                continue
            cost = re.match(r"돗테오키 게이지\s*(\d+)", head)
            cost = int(cost.group(1)) if cost else 0
            m = _STIFF_HEAD.search(head)
            stiff = int(m.group(1)) if m else 0
            for line in body.splitlines():
                if "배틀 페이즈 종료시" in line:
                    continue
                g = _GAUGE_PLUS.search(line)
                g2 = _GAUGE_OPP_FIELD.search(line)
                if (g or g2) and "상대 돗테오키" not in line:
                    self.gauge_gain.append((int(g.group(1)) if g else -int(g2.group(1)), cost, stiff))
                a = _MORE_ATTACKS.search(line)
                if a and _WHO[a.group(2)] != "self":
                    self.more_attacks.append((_WHO[a.group(2)], int(a.group(3)), cost, stiff + int(a.group(1) or 0)))


def status_turns(desc, m):
    """Turns of the status in match m of _ENEMY_STATUS: the duration words on its line (1 if none)."""
    start = desc.rfind("\n", 0, m.start()) + 1
    line = desc[start:m.end()]
    for rx, turns in _STATUS_TURNS:
        t = rx.search(line)
        if t:
            return int(t.group(1)) if turns is None else turns
    return 1


_cache = {}


def text_of(db, code):
    t = _cache.get(code)
    if t is None:
        card = db.get(code)
        t = CardText(card.desc if card else "")
        _cache[code] = t
    return t
