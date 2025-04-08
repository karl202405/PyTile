from pytile import *

# 初始化SRAM和DRAM张量
sram = SRAM()
a_dram = Tensor(Shape((4,4)), [i for i in range(16)])
b_dram = Tensor(Shape((4,4)), [i*2 for i in range(16)])

# 分配SRAM寄存器
a_reg = sram.allocate("A", Shape((4,4)))
b_reg = sram.allocate("B", Shape((4,4)))
c_reg = sram.allocate("C", Shape((4,4)))

# 执行Tile操作
load(a_dram, a_reg)    # 从DRAM加载
load(b_dram, b_reg)    # 从DRAM加载
matmul(a_reg, b_reg, c_reg)  # 矩阵乘法
store(c_reg, a_dram)   # 存回DRAM