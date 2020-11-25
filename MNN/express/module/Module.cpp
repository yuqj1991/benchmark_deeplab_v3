//
//  Module.cpp
//  MNN
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "FixModule.hpp"
#include "PipelineModule.hpp"
#include "core/FileLoader.hpp"

namespace MNN {
namespace Express {

class EmptyModule : public Module {
public:
    EmptyModule(const std::vector<Express::VARP>& parameters) {
        for (auto p : parameters) {
            addParameter(p);
        }
    }
    virtual ~EmptyModule() {
        // Do nothing
    }
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
        return {};
    }

protected:
    EmptyModule() = default;

    Module* clone(Module::CloneContext* ctx) const override {
        EmptyModule* module(new EmptyModule);
        return this->cloneBaseTo(ctx, module);
    }
};

Module* Module::createEmpty(const std::vector<Express::VARP>& parameters) {
    return new EmptyModule(parameters);
}

Express::VARP Module::forward(Express::VARP input) {
    return this->onForward({input})[0];
}
std::vector<Express::VARP> Module::parameters() const {
    std::vector<Express::VARP> result;
    _collectParameters(result);
    return result;
}
bool Module::loadParameters(const std::vector<Express::VARP>& parameters) {
    std::vector<Express::VARP> result;
    _collectParameters(result);
    if (parameters.empty() || parameters.size() != result.size()) {
        MNN_ERROR("Error parameters, empty or parameter size not match \n");
        return false;
    }
    for (int i=0; i<parameters.size(); ++i) {
        if (nullptr != result[i].get()) {
            // Check Origin parameter's size
            auto dstInfo = result[i]->getInfo();
            auto srcInfo = parameters[i]->getInfo();
            if (dstInfo->dim.size() != srcInfo->dim.size() || dstInfo->order != srcInfo->order) {
                MNN_ERROR("Error parameters %d, dim size or order not match \n", i);
                return false;
            }
            if (dstInfo->size != srcInfo->size || dstInfo->type != srcInfo->type) {
                MNN_ERROR("Error parameters %d, size or type not match \n", i);
                return false;
            }
        }
        Variable::replace(result[i], parameters[i]);
    }
    return true;
}
void Module::setIsTraining(const bool isTraining) {
    mIsTraining = isTraining;
    for (auto c : mChildren) {
        c->setIsTraining(isTraining);
    }
}

bool Module::getIsTraining() {
    return mIsTraining;
}

void Module::registerModel(const std::vector<std::shared_ptr<Module>>& children) {
    mChildren.insert(mChildren.begin(), children.begin(), children.end());
}
int Module::addParameter(VARP parameter) {
    auto res = mParameters.size();
    mParameters.emplace_back(parameter);
    return (int)res;
}

void Module::setParameter(Express::VARP parameter, int index) {
    if (index < 0 || index >= mParameters.size()) {
        MNN_ERROR("Module error: index out of range: %d - %d:\n", index, (int)mParameters.size());
        return;
    }
    mParameters[index] = parameter;
}

void Module::_collectParameters(std::vector<Express::VARP>& result) const {
    for (auto p : mParameters) {
        result.push_back(p);
    }
    for (auto c : mChildren) {
        c->_collectParameters(result);
    }
}
void Module::clearCache() {
    for (auto c : mChildren) {
        c->clearCache();
    }
    this->onClearCache();
}

Module* Module::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const char* fileName, bool dynamic) {
    AutoStorage<uint8_t> buffer;
    {
        FileLoader loader(fileName);
        if (!loader.valid()) {
            MNN_ERROR("Error for open %s\n", fileName);
            return {};
        }
        loader.read();
        if (!loader.valid()) {
            return {};
        }
        loader.merge(buffer);
        if (buffer.get() == nullptr) {
            return {};
        }
    }
    return load(inputs, outputs, buffer.get(), buffer.size(), dynamic);
}

Module* Module::load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, bool dynamic) {
    return PipelineModule::load(inputs, outputs, buffer, length, dynamic);
}

EXPRP Module::CloneContext::getOrClone(EXPRP expr) {
    auto it = mExprMap.find(expr.get());
    if (it == mExprMap.end()) {
        // EXPRP replica = expr->clone(shareParams);
        // TODO(hjchen2): Clone expr.
        EXPRP replica = expr;
        it = mExprMap.emplace(expr.get(), replica).first;
    }
    return it->second;
}

VARP Module::CloneContext::getOrClone(VARP var) {
    auto it = mVarMap.find(var.get());
    if (it != mVarMap.end()) {
        // TODO(hjchen2): Clone variable.
        VARP replica = var;
        it = mVarMap.emplace(var.get(), replica).first;
    }
    return it->second;
}

Module* Module::clone(const Module* module, const bool shareParams) {
    CloneContext context(shareParams);
    return module->clone(&context);
}

Module* Module::cloneBaseTo(CloneContext* ctx, Module* module) const {
    for (const Express::VARP& var : mParameters) {
        module->mParameters.push_back(ctx->getOrClone(var));
    }
    module->mIsTraining = mIsTraining;
    module->mName = mName;
    module->mType = mType;
    return module;
}

} // namespace Express
} // namespace MNN
