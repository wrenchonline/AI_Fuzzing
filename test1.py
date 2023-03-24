import capstone
from unicorn.x86_const import *
# 用于测试的x86汇编代码，使用了mov和lea指令
code = b"\x8B\x45\x08\x8D\x04\x40"

# 定义Capstone的指令集和模式
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
# Map memory for program code and stack
ADDRESS = 0x100000
SIZE = 0x10000
md.mem_map(ADDRESS, SIZE)
md.mem_map(0x7fffff0000, 0x1000)

md.reg_write(UC_X86_REG_RSP, ADDRESS + SIZE - 0x100)
md.reg_write(UC_X86_REG_RIP, ADDRESS)

# 迭代每条指令并打印寄存器和内存地址
for i in md.disasm(code, 0x1000):
    # 提取指令中涉及的寄存器
    regs = list(filter(lambda x: x.startswith('reg'), i.regs_access()))
    if len(regs) > 0:
        print(f"Instruction: {i.mnemonic} {i.op_str}")
        print(f"Registers: {', '.join(regs)}")

    # 提取指令中涉及的内存地址
    if i.mem_op:
        print(f"Memory Address: 0x{i.mem_op}")
