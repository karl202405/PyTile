from pytile.tile import *

def test_matmul():
    sram = SRAM()
    a = Tensor(Shape((2,2)), [1,2,3,4])
    b = Tensor(Shape((2,2)), [1,0,0,1])
    c = Tensor(Shape((2,2)), [0]*4)
    
    a_reg = sram.allocate("A", Shape((2,2)))
    b_reg = sram.allocate("B", Shape((2,2)))
    c_reg = sram.allocate("C", Shape((2,2)))
    
    load(a, a_reg)
    load(b, b_reg)
    matmul(a_reg, b_reg, c_reg)
    store(c_reg, c)
    
    assert c._data == [1,2,3,4]  # 单位矩阵乘法

def test_dlt():
    sram = SRAM()
    src = Tensor(Shape((2,2)), [1,2,3,4])
    dst = Tensor(Shape((2,2)), [0]*4)
    reg1 = sram.allocate("R1", Shape((2,2)))
    reg2 = sram.allocate("R2", Shape((2,2)))
    
    load(src, reg1)
    dlt(reg1, reg2)
    store(reg2, dst)
    
    assert dst._data == [1,3,2,4]  # 转置结果