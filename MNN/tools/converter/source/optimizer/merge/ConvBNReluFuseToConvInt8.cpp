//
//  ConvBNReluFuseToConvInt8.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

static auto gRegister = []() {
    auto match = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_FloatToInt8) {
            return false;
        }

        VARP convert1_var   = expr->inputs().at(0);
        EXPRP convert1_expr = convert1_var->expr().first;
        if (!convert1_expr->get() || convert1_expr->get()->type() != OpType_ConvertTensor) {
            return false;
        }

        VARP conv_var   = convert1_expr->inputs().at(0);
        EXPRP conv_expr = conv_var->expr().first;
        if (!conv_expr->get() || (conv_expr->get()->type() != OpType_Convolution &&
                                  conv_expr->get()->type() != OpType_ConvolutionDepthwise)) {
            return false;
        }

        VARP convert2_var   = conv_expr->inputs().at(0);
        EXPRP convert2_expr = convert2_var->expr().first;
        if (!convert2_expr->get() || convert2_expr->get()->type() != OpType_ConvertTensor) {
            return false;
        }

        VARP quant_var   = convert2_expr->inputs().at(0);
        EXPRP quant_expr = quant_var->expr().first;
        if (!quant_expr->get() || quant_expr->get()->type() != OpType_Int8ToFloat) {
            return false;
        }
        return true;
    };

    auto transform = [](EXPRP expr) {
        auto convert1          = expr->inputs()[0];
        auto convert1_expr     = convert1->expr().first;
        auto convOutput        = convert1_expr->inputs()[0];
        auto convExpr          = convOutput->expr().first;
        auto convert2          = convExpr->inputs()[0];
        auto convert2_expr     = convert2->expr().first;
        auto int8ToFloatOutput = convert2_expr->inputs()[0];
        auto int8ToFloatExpr   = int8ToFloatOutput->expr().first;
        auto convInt8Input     = int8ToFloatExpr->inputs()[0];

        std::unique_ptr<OpT> convOp(convExpr->get()->UnPack());
        auto convParams  = convOp->main.AsConvolution2D();
        auto weightFloat = convParams->weight;
        auto biasFloat   = convParams->bias;
        auto& common     = convParams->common;

        std::unique_ptr<OpT> int8ToFloatOp(int8ToFloatExpr->get()->UnPack());
        float inputScale = int8ToFloatOp->main.AsQuantizedFloatParam()->tensorScale[0];

        std::unique_ptr<OpT> floatToInt8Op(expr->get()->UnPack());
        float outputScale = 1.f / floatToInt8Op->main.AsQuantizedFloatParam()->tensorScale[0];

        std::vector<int8_t> int8Weight;
        std::vector<int32_t> int32Bias;
        std::vector<float> scale;

        const int ko = common->outputCount;
        const int ki = common->inputCount / common->group;
        const int kh = common->kernelY;
        const int kw = common->kernelX;

        VARP weightVar      = _Const(weightFloat.data(), {ko, ki, kh, kw}, NCHW);
        VARP biasVar        = _Const(biasFloat.data(), {ko, 1, 1, 1}, NCHW);
        VARP inputScaleVar  = _Const(inputScale, {}, NCHW);
        VARP outputScaleVar = _Const(outputScale, {}, NCHW);

        int nbits       = int8ToFloatOp->main.AsQuantizedFloatParam()->nbits;
        float max_value = (1U << (nbits - 1)) - 1;
        // Lower bitwidths (< 8bits) is only used by winograd-aware optimization.
        // For winograd-aware, activation has two quantization bitwidths,
        //   - 7bits:
        //     Due to fewer accumulation times, 7 bits can be used for Conv1xN.
        //     In this case, the weight should be limited to 42 in order to
        //     prevent calculation overflow.
        //   - 6bits
        //     For general Conv3x3, only 6 bits can be satisfied, and the weight
        //     should be limited to 15 in order to prevent overflow.
        if (nbits == 7) {
            max_value = 42;
        }
        if (nbits == 6) {
            max_value = 15;
        }
        auto weightScale = _Maximum(_ReduceMax(_Abs(weightVar), {1, 2, 3}, true), _Scalar<float>(1e-6)) *
                           _Scalar<float>(1.f / max_value);
        auto quanWeight = _Cast<int8_t>(_Round(weightVar * _Reciprocal(weightScale)));
        auto convScale  = _Reshape(_Reciprocal(outputScaleVar), {-1, 1, 1, 1}) * weightScale * inputScaleVar;
        auto quanBias   = _Cast<int32_t>(biasVar * _Reciprocal(weightScale * inputScaleVar));

        {
            auto info = quanWeight->getInfo();
            int8Weight.resize(info->size);
            auto ptr = quanWeight->readMap<int8_t>();
            ::memcpy(int8Weight.data(), ptr, int8Weight.size() * sizeof(int8_t));
        }
        {
            auto biasinfo = quanBias->getInfo();
            int32Bias.resize(biasinfo->size);
            auto ptr = quanBias->readMap<int32_t>();
            ::memcpy(int32Bias.data(), ptr, int32Bias.size() * sizeof(int32_t));

            auto info = convScale->getInfo();
            scale.resize(info->size);
            MNN_ASSERT(scale.size() == int32Bias.size());
            auto ptrScale = convScale->readMap<float>();
            ::memcpy(scale.data(), ptrScale, scale.size() * sizeof(float));
        }

        std::unique_ptr<Convolution2DT> conv(new MNN::Convolution2DT);
        conv->common.reset(new MNN::Convolution2DCommonT);
        auto* conv_common        = conv->common.get();
        conv_common->relu        = common->relu || common->relu6;
        conv_common->group       = common->group;
        conv_common->outputCount = common->outputCount;
        conv_common->inputCount  = common->inputCount;
        conv_common->kernelX     = kw;
        conv_common->kernelY     = kh;
        conv_common->padX        = common->padX;
        conv_common->padY        = common->padY;
        conv_common->dilateX     = common->dilateX;
        conv_common->dilateY     = common->dilateY;
        conv_common->strideX     = common->strideX;
        conv_common->strideY     = common->strideY;
        conv_common->padMode     = common->padMode;

        MNN_ASSERT(int8Weight.size() == common->inputCount * (common->outputCount / common->group) * kw * kh);

        conv->symmetricQuan.reset(new QuantizedFloatParamT);

        bool is_depthwise = common->inputCount == common->outputCount && common->outputCount == common->group;

        conv->symmetricQuan->bias   = std::move(int32Bias);
        conv->symmetricQuan->scale  = std::move(scale);
        conv->symmetricQuan->weight = std::move(int8Weight);
        conv->symmetricQuan->nbits  = nbits;

        std::unique_ptr<OpT> conv_op(new OpT);
        conv_op->name = expr->name();
        conv_op->type = OpType_ConvInt8;
        if (is_depthwise) {
            conv_op->type = OpType_DepthwiseConvInt8;
        }
        conv_op->main.type  = OpParameter_Convolution2D;
        conv_op->main.value = conv.release();

        auto conv_expr = Expr::create(conv_op.get(), {convInt8Input});
        conv_expr->setName(convExpr->name());
        auto conv_var = Variable::create(conv_expr);
        conv_var->setName(convExpr->outputName(0));
        Expr::replace(expr, conv_expr);
        return true;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("ConvBNReluFuseToConvInt8", match, transform,
                                                       PASS_PRIORITY_MIDDLE);
    return true;
}();

}
} // namespace MNN
