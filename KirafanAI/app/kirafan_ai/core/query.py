"""Parsing of OCG_DuelQuery* buffers."""
import struct

from .constants import *  # noqa: F401,F403
from .reader import Reader

VIEW_QUERY_FLAGS = (QUERY_CODE | QUERY_POSITION | QUERY_TYPE | QUERY_LEVEL | QUERY_RANK | QUERY_ATTRIBUTE |
                    QUERY_RACE | QUERY_ATTACK | QUERY_DEFENSE | QUERY_BASE_ATTACK | QUERY_BASE_DEFENSE |
                    QUERY_OVERLAY_CARD | QUERY_COUNTERS | QUERY_OWNER | QUERY_STATUS | QUERY_IS_PUBLIC)


def parse_card(r):
    """Parse one card's info block. Returns dict, or None for an empty slot."""
    size = r.u16()
    if size == 0:
        return None
    card = {}
    while True:
        end = r.pos + size
        flag = r.u32()
        if flag == QUERY_END:
            r.pos = end
            break
        if flag == QUERY_CODE:
            card["code"] = r.u32()
        elif flag == QUERY_POSITION:
            card["pos"] = r.u32()
        elif flag == QUERY_ALIAS:
            card["alias"] = r.u32()
        elif flag == QUERY_TYPE:
            card["type"] = r.u32()
        elif flag == QUERY_LEVEL:
            card["level"] = r.u32()
        elif flag == QUERY_RANK:
            card["rank"] = r.u32()
        elif flag == QUERY_ATTRIBUTE:
            card["attribute"] = r.u32()
        elif flag == QUERY_RACE:
            card["race"] = r.u64()
        elif flag == QUERY_ATTACK:
            card["atk"] = r.i32()
        elif flag == QUERY_DEFENSE:
            card["def"] = r.i32()
        elif flag == QUERY_BASE_ATTACK:
            card["base_atk"] = r.i32()
        elif flag == QUERY_BASE_DEFENSE:
            card["base_def"] = r.i32()
        elif flag == QUERY_OVERLAY_CARD:
            n = r.u32()
            card["overlay"] = [r.u32() for _ in range(n)]
        elif flag == QUERY_COUNTERS:
            n = r.u32()
            counters = {}
            for _ in range(n):
                v = r.u32()
                counters[v & 0xffff] = v >> 16
            card["counters"] = counters
        elif flag == QUERY_OWNER:
            card["owner"] = r.u8()
        elif flag == QUERY_STATUS:
            card["status"] = r.u32()
        elif flag == QUERY_IS_PUBLIC:
            card["public"] = r.u8()
        r.pos = end
        size = r.u16()
    return card


# Fast path for VIEW_QUERY_FLAGS: card::get_infos writes the requested fields in a fixed order,
# each as [u16 size][u32 flag][value].
_FIXED = struct.Struct("<" + "HII" * 6 + "HIQ" + "HIi" * 4)
_FIXED_FLAGS = (QUERY_CODE, QUERY_POSITION, QUERY_TYPE, QUERY_LEVEL, QUERY_RANK, QUERY_ATTRIBUTE, QUERY_RACE,
                QUERY_ATTACK, QUERY_DEFENSE, QUERY_BASE_ATTACK, QUERY_BASE_DEFENSE)
_HDR = struct.Struct("<HII")
_U8F = struct.Struct("<HIB")
_U32 = struct.Struct("<I")


def _parse_card_fast(buf, pos):
    v = _FIXED.unpack_from(buf, pos)
    if tuple(v[1::3]) != _FIXED_FLAGS:
        return None, pos
    pos += _FIXED.size
    card = {"code": v[2], "pos": v[5], "type": v[8], "level": v[11], "rank": v[14], "attribute": v[17],
            "race": v[20], "atk": v[23], "def": v[26], "base_atk": v[29], "base_def": v[32]}
    _, flag, n = _HDR.unpack_from(buf, pos)
    if flag != QUERY_OVERLAY_CARD:
        return None, pos
    pos += _HDR.size + 4 * n
    card["overlay"] = [0] * n
    _, flag, n = _HDR.unpack_from(buf, pos)
    if flag != QUERY_COUNTERS:
        return None, pos
    pos += _HDR.size
    counters = {}
    for i in range(n):
        x = _U32.unpack_from(buf, pos + 4 * i)[0]
        counters[x & 0xffff] = x >> 16
    pos += 4 * n
    card["counters"] = counters
    _, flag, owner = _U8F.unpack_from(buf, pos)
    if flag != QUERY_OWNER:
        return None, pos
    pos += _U8F.size
    _, flag, status = _HDR.unpack_from(buf, pos)
    if flag != QUERY_STATUS:
        return None, pos
    pos += _HDR.size
    _, flag, public = _U8F.unpack_from(buf, pos)
    if flag != QUERY_IS_PUBLIC:
        return None, pos
    pos += _U8F.size
    card["owner"] = owner
    card["status"] = status
    card["public"] = public
    # remaining blocks up to QUERY_END
    while True:
        size, flag = struct.unpack_from("<HI", buf, pos)
        pos += 2 + size
        if flag == QUERY_END:
            return card, pos


def parse_location(buf):
    r = Reader(buf)
    total = r.u32()
    end = 4 + total
    cards = []
    while r.pos < end:
        if buf[r.pos] == 0 and buf[r.pos + 1] == 0:
            r.pos += 2
            cards.append(None)
            continue
        card, pos = _parse_card_fast(buf, r.pos)
        if card is not None:
            r.pos = pos
            cards.append(card)
        else:
            cards.append(parse_card(r))
    return cards


def parse_field(buf):
    r = Reader(buf)
    out = {"options": r.u32(), "players": []}
    for _ in range(2):
        p = {"lp": r.u32(), "mzone": [], "szone": []}
        for zone, n in (("mzone", 7), ("szone", 8)):
            for _ in range(n):
                if r.u8():
                    p[zone].append((r.u8(), r.u32()))
                else:
                    p[zone].append(None)
        p["deck"] = r.u32()
        p["hand"] = r.u32()
        p["grave"] = r.u32()
        p["removed"] = r.u32()
        p["extra"] = r.u32()
        p["extra_p"] = r.u32()
        out["players"].append(p)
    chain = []
    for _ in range(r.u32()):
        code = r.u32()
        con, loc, seq, pos = r.loc_info()
        tcon = r.u8()
        r.u8()
        r.u32()
        desc = r.u64()
        chain.append({"code": code, "con": con, "loc": loc, "seq": seq, "trig_con": tcon, "desc": desc})
    out["chain"] = chain
    return out
