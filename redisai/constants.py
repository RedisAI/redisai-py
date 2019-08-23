from enum import Enum


class Device(Enum):
    cpu = 'CPU'
    gpu = 'GPU'


class Backend(Enum):
    tf = 'TF'
    torch = 'TORCH'
    onnx = 'ONNX'


class DType(Enum):
    float = 'FLOAT'
    double = 'DOUBLE'
    int8 = 'INT8'
    int16 = 'INT16'
    int32 = 'INT32'
    int64 = 'INT64'
    uint8 = 'UINT8'
    uint16 = 'UINT16'
    uint32 = 'UINT32'
    uint64 = 'UINT64'

    # aliases
    float32 = 'FLOAT'
    float64 = 'DOUBLE'