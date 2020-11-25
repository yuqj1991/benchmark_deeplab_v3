# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class TfQuantizedConv2D(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsTfQuantizedConv2D(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TfQuantizedConv2D()
        x.Init(buf, n + offset)
        return x

    # TfQuantizedConv2D
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # TfQuantizedConv2D
    def Bias(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # TfQuantizedConv2D
    def BiasAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # TfQuantizedConv2D
    def BiasLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TfQuantizedConv2D
    def Biasflag(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # TfQuantizedConv2D
    def Common(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Convolution2DCommon import Convolution2DCommon
            obj = Convolution2DCommon()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # TfQuantizedConv2D
    def Weight(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # TfQuantizedConv2D
    def WeightAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # TfQuantizedConv2D
    def WeightLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TfQuantizedConv2D
    def ActivationType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # TfQuantizedConv2D
    def Multiplier(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # TfQuantizedConv2D
    def OutMax(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # TfQuantizedConv2D
    def OutMin(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # TfQuantizedConv2D
    def Shift(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # TfQuantizedConv2D
    def BiasQuantizedParam(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .QuantizedParam import QuantizedParam
            obj = QuantizedParam()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # TfQuantizedConv2D
    def DepthMultiplier(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # TfQuantizedConv2D
    def FilterQuantizedParam(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .QuantizedParam import QuantizedParam
            obj = QuantizedParam()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # TfQuantizedConv2D
    def InputQuantizedParam(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .QuantizedParam import QuantizedParam
            obj = QuantizedParam()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # TfQuantizedConv2D
    def ModelFormat(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # TfQuantizedConv2D
    def OutputQuantizedParam(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(32))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .QuantizedParam import QuantizedParam
            obj = QuantizedParam()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def TfQuantizedConv2DStart(builder): builder.StartObject(15)
def TfQuantizedConv2DAddBias(builder, bias): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(bias), 0)
def TfQuantizedConv2DStartBiasVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def TfQuantizedConv2DAddBiasflag(builder, biasflag): builder.PrependBoolSlot(1, biasflag, 0)
def TfQuantizedConv2DAddCommon(builder, common): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(common), 0)
def TfQuantizedConv2DAddWeight(builder, weight): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(weight), 0)
def TfQuantizedConv2DStartWeightVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def TfQuantizedConv2DAddActivationType(builder, activationType): builder.PrependInt8Slot(4, activationType, 0)
def TfQuantizedConv2DAddMultiplier(builder, multiplier): builder.PrependInt32Slot(5, multiplier, 0)
def TfQuantizedConv2DAddOutMax(builder, outMax): builder.PrependInt32Slot(6, outMax, 0)
def TfQuantizedConv2DAddOutMin(builder, outMin): builder.PrependInt32Slot(7, outMin, 0)
def TfQuantizedConv2DAddShift(builder, shift): builder.PrependInt32Slot(8, shift, 0)
def TfQuantizedConv2DAddBiasQuantizedParam(builder, biasQuantizedParam): builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(biasQuantizedParam), 0)
def TfQuantizedConv2DAddDepthMultiplier(builder, depthMultiplier): builder.PrependInt32Slot(10, depthMultiplier, 0)
def TfQuantizedConv2DAddFilterQuantizedParam(builder, filterQuantizedParam): builder.PrependUOffsetTRelativeSlot(11, flatbuffers.number_types.UOffsetTFlags.py_type(filterQuantizedParam), 0)
def TfQuantizedConv2DAddInputQuantizedParam(builder, inputQuantizedParam): builder.PrependUOffsetTRelativeSlot(12, flatbuffers.number_types.UOffsetTFlags.py_type(inputQuantizedParam), 0)
def TfQuantizedConv2DAddModelFormat(builder, modelFormat): builder.PrependInt8Slot(13, modelFormat, 0)
def TfQuantizedConv2DAddOutputQuantizedParam(builder, outputQuantizedParam): builder.PrependUOffsetTRelativeSlot(14, flatbuffers.number_types.UOffsetTFlags.py_type(outputQuantizedParam), 0)
def TfQuantizedConv2DEnd(builder): return builder.EndObject()
