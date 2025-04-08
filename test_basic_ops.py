from pytile.tile import *

def test_load_store():
    sram = SRAM()
    dram = Tensor(Shape((2,2)), [1,2,3,4])
    reg = sram.allocate("R1", Shape((2,2)))
    
    load(dram, reg)
    assert reg._data == [1,2,3,4]
    
    store(reg, dram)
    assert dram._data == [1,2,3,4]

def test_scalar_ops():
    sram = SRAM()
    src = Tensor(Shape((2,2)), [1,1,1,1])
    dst = Tensor(Shape((2,2)), [0]*4)
    reg = sram.allocate("R1", Shape((2,2)))
    
    load(src, reg)
    scalar_op(reg, 5.0, reg)
    store(reg, dst)
    assert dst._data == [5,5,5,5]