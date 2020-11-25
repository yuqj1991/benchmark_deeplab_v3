# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class PackParam(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsPackParam(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PackParam()
        x.Init(buf, n + offset)
        return x

    # PackParam
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # PackParam
    def DataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # PackParam
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def PackParamStart(builder): builder.StartObject(2)
def PackParamAddDataType(builder, dataType): builder.PrependInt32Slot(0, dataType, 0)
def PackParamAddAxis(builder, axis): builder.PrependInt32Slot(1, axis, 0)
def PackParamEnd(builder): return builder.EndObject()
