from pytile.tile import *

def test_vec_add():
    sram = SRAM()
    v1 = Tensor(Shape((1,4)), [1,2,3,4])
    v2 = Tensor(Shape((1,4)), [4,3,2,1])
    v3 = Tensor(Shape((1,4)), [0]*4)
    
    r1 = sram.allocate("V1", Shape((1,4)))
    r2 = sram.allocate("V2", Shape((1,4))) 
    r3 = sram.allocate("V3", Shape((1,4)))
    
    load(v1, r1)
    load(v2, r2)
    vec_add(r1, r2, r3)
    store(r3, v3)
    
    assert v3._data == [5,5,5,5]

def test_composite_ops():
    sram = SRAM()
    v = Tensor(Shape((1,4)), [1,1,1,1])
    s = 2.0
    out = Tensor(Shape((1,4)), [0]*4)
    
    reg = sram.allocate("R", Shape((1,4)))
    tmp = sram.allocate("TMP", Shape((1,4)))
    
    load(v, reg)
    scalar_op(reg, s, tmp)
    vec_add(reg, tmp, reg)  # v + (v * s)
    store(reg, out)
    
    assert out._data == [3,3,3,3]