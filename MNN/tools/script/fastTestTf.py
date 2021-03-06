#!/usr/bin/python
import os
import sys
import numpy as np
import tensorflow as tf

class TestModel():
    def __copy_to_here(self, modelName):
        newModel = 'tf/test.pb'
        print(os.popen("mkdir tf").read())
        print(os.popen("cp " + modelName + ' ' + newModel).read())
        self.modelName = newModel
        self.model = self.__load_graph(self.modelName)
        self.inputOps, self.outputOps = self.__analyze_inputs_outputs(self.model)
        self.outputs = [output.name for output in self.outputOps]
    def __init__(self, modelName):
        self.__copy_to_here(modelName)
    def __run_mnn(self):
        result = os.popen("./TestConvertResult Tf tf").read()
        print(result)
        return result
    def __load_graph(self, filename):
        f = tf.io.gfile.GFile(filename, "rb")
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        graph = tf.compat.v1.get_default_graph() 
        return graph
    def __analyze_inputs_outputs(self, graph):
        ops = graph.get_operations()
        outputs_set = set(ops)
        inputs = []
        for op in ops:
            if len(op.inputs) == 0 and op.type != 'Const':
                inputs.append(op)
            else:
                for input_tensor in op.inputs:
                    if input_tensor.op in outputs_set:
                        outputs_set.remove(input_tensor.op)
        outputs = [op for op in outputs_set if op.type != 'Assert']
        return (inputs, outputs)
    def __get_shape(self, op):
        shape = list(op.outputs[0].shape)
        for i in range(len(shape)):
            if shape[i] == None:
                shape[i] = 299
        return shape
    def __run_tf(self):
        jsonDict = {}
        jsonDict['inputs'] = []
        jsonDict['outputs'] = []
        inputs = {}
        print(self.modelName)
        for inputVar in self.inputOps:
            inp = {}
            inp['name'] = inputVar.name
            inp['shape'] = self.__get_shape(inputVar)
            inputs[inputVar.name + ':0'] = np.random.uniform(0, 255, inp['shape']).astype(np.typeDict[inputVar.outputs[0].dtype.name])
            jsonDict['inputs'].append(inp)
        print([output.name for output in self.outputOps])
        for output in self.outputOps:
            jsonDict['outputs'].append(output.name)

        import json
        jsonString = json.dumps(jsonDict, indent=4)
        with open('tf/input.json', 'w') as f:
            f.write(jsonString)

        print('inputs:')
        for key in inputs:
            print(key)
            f = open("tf/" + key[:-2] + '.txt', 'w')
            np.savetxt(f, inputs[key].flatten())
            f.close()
        sess = tf.compat.v1.Session()
        outputs_tensor = [(output + ':0') for output in self.outputs]
        outputs = sess.run(outputs_tensor, inputs)
        print('outputs:')
        for i in range(len(outputs)):
            outputName = self.outputs[i]
            name = 'tf/' + outputName + '.txt'
            # print(name, outputs[i].shape)
            f = open(name, 'w')
            np.savetxt(f, outputs[i].flatten())
            f.close()
    def Test(self):
        self.__run_tf()
        res = self.__run_mnn()
        return res

if __name__ == '__main__':
    modelName = sys.argv[1]
    t = TestModel(modelName)
    t.Test()
