"""Card database access (expansions/kirafan.cdb)."""
import sqlite3

from . import paths
from .core.constants import (RACE_WARRIOR, RACE_SPELLCASTER, RACE_FAIRY, RACE_FIEND, RACE_ZOMBIE,
                             TYPE_TOKEN, TYPE_XYZ, SUMMONER_CODES, TEST_SUMMONER)

CLASS_RACES = [RACE_WARRIOR, RACE_SPELLCASTER, RACE_FAIRY, RACE_FIEND, RACE_ZOMBIE]
CLASS_NAMES = {RACE_WARRIOR: "전사", RACE_SPELLCASTER: "마법사", RACE_FAIRY: "나이트",
               RACE_FIEND: "승려", RACE_ZOMBIE: "알케미스트"}
ATTRIBUTES = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40]
ATTRIBUTE_NAMES = {0x01: "땅", 0x02: "물", 0x04: "불", 0x08: "바람", 0x10: "해", 0x20: "달", 0x40: "소환사"}
# Cards that count as one card: at most 3 copies of them together in a deck, one entry in the pool and the
# card rankings (under the key code, the one generated decks hold). alias -> key: 이상한 나라의 앨리스 카터릿 ->
# 다크니스 앨리스 카터릿 (the better one early in a game)
CARD_ALIASES = {10055113: 10055112}


def canonical(code):
    return CARD_ALIASES.get(code, code)


class Card:
    __slots__ = ("code", "alias", "setcodes", "type", "level", "attribute", "race", "attack",
                 "defense", "name", "desc", "strings")

    def __init__(self, row, text):
        code, alias, setcode, ctype, atk, dfn, level, race, attribute = row
        self.code = code
        self.alias = alias
        self.setcodes = [(setcode >> (16 * i)) & 0xffff for i in range(4) if (setcode >> (16 * i)) & 0xffff]
        self.type = ctype
        self.level = level & 0xff
        self.attribute = attribute
        self.race = race
        self.attack = atk
        self.defense = dfn
        self.name = text[0] if text else str(code)
        self.desc = text[1] if text else ""
        self.strings = list(text[2:18]) if text else [""] * 16

    @property
    def is_battle(self):
        return bool(self.race & (RACE_WARRIOR | RACE_SPELLCASTER | RACE_FAIRY))

    @property
    def is_support(self):
        return bool(self.race & (RACE_FIEND | RACE_ZOMBIE))

    @property
    def is_creamate(self):
        return (self.is_battle or self.is_support) and not (self.type & (TYPE_TOKEN | TYPE_XYZ))

    @property
    def class_index(self):
        for i, r in enumerate(CLASS_RACES):
            if self.race & r:
                return i
        return len(CLASS_RACES)

    @property
    def attribute_index(self):
        for i, a in enumerate(ATTRIBUTES):
            if self.attribute & a:
                return i
        return len(ATTRIBUTES)


class CardDB:
    def __init__(self, path=paths.CARD_DB):
        con = sqlite3.connect(path)
        try:
            texts = {}
            for r in con.execute("select * from texts"):
                texts[r[0]] = list(r[1:]) + [""] * (18 - len(r[1:]))
            self.cards = {}
            for r in con.execute("select id, alias, setcode, type, atk, def, level, race, attribute from datas"):
                self.cards[r[0]] = Card(r, texts.get(r[0]))
        finally:
            con.close()
        self.codes = sorted(self.cards)
        self.set_vocab_order(self.codes)

    def set_vocab_order(self, order):
        """Embedding ids follow `order` (the order saved in a checkpoint); codes added to the cdb
        later are appended, so existing ids never shift when new cards are created."""
        known = set(order)
        # cards deleted from the cdb keep their slot so later ids do not move either
        self.vocab_codes = list(order) + [c for c in sorted(self.cards) if c not in known]
        # index 0 = unknown/hidden card. Used for effect descriptions too, where each summoner card
        # stores different system strings, so every code keeps its own id here.
        self.vocab = {c: i + 1 for i, c in enumerate(self.vocab_codes)}
        # Card identity: every summoner has the same effects, so they share one id
        self.card_vocab = dict(self.vocab)
        summoners = [c for c in SUMMONER_CODES if c in self.vocab]
        for c in summoners:
            self.card_vocab[c] = self.vocab[summoners[0]]

    def __getitem__(self, code):
        return self.cards[code]

    def get(self, code):
        return self.cards.get(code)

    def creamate_pool(self):
        """Cards that may be put in a main deck (an alias counts as its key card, CARD_ALIASES)."""
        return [c for c in self.codes if self.cards[c].is_creamate and c not in CARD_ALIASES]

    def summoner_pool(self):
        return [c for c in SUMMONER_CODES if c in self.cards]

    def describe(self, desc):
        """Human-readable text for an effect description value (aux.Stringid)."""
        code, idx = desc >> 20, desc & 0xfffff
        card = self.cards.get(code)
        if card and idx < 16 and card.strings[idx]:
            return card.strings[idx].replace("\n", " ")
        if code == 0:
            return "system:%d" % idx
        return "%s#%d" % (card.name if card else code, idx)

    def name(self, code):
        card = self.cards.get(code)
        return card.name if card else str(code)


_db = None


def get_db():
    global _db
    if _db is None:
        _db = CardDB()
    return _db


__all__ = ["CardDB", "Card", "get_db", "CLASS_NAMES", "ATTRIBUTE_NAMES", "TEST_SUMMONER"]
