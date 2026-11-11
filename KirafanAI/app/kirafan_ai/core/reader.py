import struct

_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_I16 = struct.Struct("<h")
_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_U64 = struct.Struct("<Q")


class Reader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf, pos=0):
        self.buf = buf
        self.pos = pos

    def remaining(self):
        return len(self.buf) - self.pos

    def u8(self):
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def u16(self):
        v = _U16.unpack_from(self.buf, self.pos)[0]
        self.pos += 2
        return v

    def i16(self):
        v = _I16.unpack_from(self.buf, self.pos)[0]
        self.pos += 2
        return v

    def u32(self):
        v = _U32.unpack_from(self.buf, self.pos)[0]
        self.pos += 4
        return v

    def i32(self):
        v = _I32.unpack_from(self.buf, self.pos)[0]
        self.pos += 4
        return v

    def u64(self):
        v = _U64.unpack_from(self.buf, self.pos)[0]
        self.pos += 8
        return v

    def skip(self, n):
        self.pos += n

    def loc_info(self):
        """controler u8, location u8, sequence u32, position u32"""
        con = self.u8()
        loc = self.u8()
        seq = self.u32()
        pos = self.u32()
        return con, loc, seq, pos


def split_messages(buf):
    """Core buffers are a sequence of [u32 length][message bytes]."""
    out = []
    pos = 0
    n = len(buf)
    while pos + 4 <= n:
        size = _U32.unpack_from(buf, pos)[0]
        pos += 4
        out.append(bytes(buf[pos:pos + size]))
        pos += size
    return out


def pack_i32(*vals):
    return struct.pack("<%di" % len(vals), *vals)
