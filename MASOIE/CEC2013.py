import ctypes
import numpy

# lib = ctypes.cdll.LoadLibrary('../CEC2013-distributed/libCEC2013.so')
lib = ctypes.cdll.LoadLibrary('./CEC2013-distributed/libCEC2013.so')
lib.getMinX.restype = ctypes.c_double
lib.getMaxX.restype = ctypes.c_double
lib.local_eva.restype = ctypes.c_double
lib.global_eva.restype = ctypes.c_double
lib.getNetworkGraph.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))

class CEC2013(object):
    def __init__(self,id):
        lib.init(convert_type(id))
        self.getDimension = lib.getDimension
        self.getGroupNum = lib.getGroupNum
        self.getMinX = lib.getMinX
        self.getMaxX = lib.getMaxX
    def local_eva(self,x,group_index):
        return lib.local_eva(convert_type(x),group_index)
    def global_eva(self,x):
        return lib.global_eva(convert_type(x))
    def getNetworkGraph(self):
        graph = lib.getNetworkGraph()
        len = lib.getGroupNum()
        res = [ [graph[i][j] for j in range(len)] for i in range(len)]
        return res


def convert_type(input):
    ctypes_map = {int:ctypes.c_int,
              float:ctypes.c_double,
              numpy.float64:ctypes.c_double,
              numpy.float32:ctypes.c_double,
              str:ctypes.c_char_p
              }
    input_type = type(input)
    if input_type is list or input_type is numpy.ndarray:
        length = len(input)
        if length==0:
            print("convert type failed...input is "+input)
            return None
        else:
            arr = (ctypes_map[type(input[0])] * length)()
            for i in range(length):
                arr[i] = bytes(input[i],encoding="utf-8") if (type(input[0]) is str) else input[i]
            return arr
    else:
        if input_type in ctypes_map:
            return ctypes_map[input_type](bytes(input,encoding="utf-8") if type(input) is str else input)
        else:
            print("convert type failed...input is "+input)
            return None

if __name__ == '__main__':
    # t = CEC2013("100D20n3dheter-2")
    t = CEC2013("50D20n1d-6")
    print(t.getDimension())
    # print(t.getGroupNum())
    x = [10.0]*100
    # x = numpy.zeros(100)
    print(t.global_eva(x))
    print(t.local_eva(x,0))
    print(t.local_eva(x,1))
    print(t.local_eva(x,2))
    # print(numpy.array(t.getNetworkGraph()))
    # id = convert_type("100D60n3dheter-2")
    # lib.init(id)
    # print(lib.getDimension())
    # print(lib.getMinX())
    # g = lib.getNetworkGraph()
    # for i in range(20):
    #     for j in range(20):
    #         print(g[i][j])
    # print(numpy.frombuffer(g[0]))