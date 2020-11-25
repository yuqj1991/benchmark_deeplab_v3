//
//  MetalBinary.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalBinary_hpp
#define MetalBinary_hpp

#import "core/Execution.hpp"
#import "MetalDefine.h"
#include <string>
#if MNN_METAL_ENABLED
namespace MNN {

class MetalBinary : public Execution {
public:
    MetalBinary(Backend *backend, std::string type);
    virtual ~MetalBinary() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::string mKernelName;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalBinary_hpp */
