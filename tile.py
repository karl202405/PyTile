from dataclasses import dataclass
from typing import Any, Tuple, Union

@dataclass
class Shape:
    dims: Tuple[int, ...]

class Tensor:
    def __init__(self, shape: Shape, data: Any = None):
        self.shape = shape
        self._data = data if data is not None else [0] * (shape.dims[0] * shape.dims[1])
    
    def __repr__(self):
        return f"Tensor(shape={self.shape})"

class SRAM:
    """模拟SRAM中的张量寄存器"""
    def __init__(self, size: int = 1024):
        self.registers = {}
        self.size = size
    
    def allocate(self, name: str, shape: Shape) -> Tensor:
        """分配张量寄存器"""
        if name in self.registers:
            raise ValueError(f"Register {name} already allocated")
        self.registers[name] = Tensor(shape)
        return self.registers[name]

# 核心运算装饰器
def tile_op(func):
    def wrapper(*args, **kwargs):
        print(f"[TileOP] Executing {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# 基本运算实现
@tile_op
def load(src: Tensor, dest: Tensor) -> None:
    """从DRAM加载到SRAM寄存器"""
    dest._data = src._data.copy()

@tile_op
def store(src: Tensor, dest: Tensor) -> None:
    """从SRAM寄存器存回DRAM"""
    dest._data = src._data.copy()

@tile_op
def matmul(A: Tensor, B: Tensor, C: Tensor) -> None:
    """矩阵乘法 (MxN) * (NxK) -> (MxK)"""
    m, n = A.shape.dims
    n_, k = B.shape.dims
    assert n == n_, "Matrix dimensions mismatch"
    
    for i in range(m):
        for j in range(k):
            C._data[i*k + j] = sum(
                A._data[i*n + x] * B._data[x*k + j] for x in range(n)
            )

@tile_op
def dlt(src: Tensor, dest: Tensor) -> None:
    """数据布局转换 (Data Layout Transform)"""
    # 简化的转置操作
    m, n = src.shape.dims
    for i in range(m):
        for j in range(n):
            dest._data[j*m + i] = src._data[i*n + j]

@tile_op
def vec_add(A: Tensor, B: Tensor, C: Tensor) -> None:
    """向量加法"""
    assert A.shape.dims == B.shape.dims == C.shape.dims
    for i in range(len(A._data)):
        C._data[i] = A._data[i] + B._data[i]

@tile_op
def scalar_op(A: Tensor, scalar: float, C: Tensor) -> None:
    """标量运算"""
    for i in range(len(A._data)):
        C._data[i] = A._data[i] * scalar