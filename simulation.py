from unicorn import *
from unicorn.x86_const import *
from keystone import *
import capstone

#fuzzing + ai

Main = """
_start:
    sub rsp, 100
    mov rax, 0x190000
    push rax
    push 12
    call overflow
    add rsp, 100
    mov rax, 0
    ret
overflow:
    push rbp
    mov rbp, rsp
    sub rsp, rsi
    mov r8,rsi
    mov rsi, rax
    lea rdi, [rsp]
    MOV rcx, 22
    rep movsb
    add rsp, r8
    mov rsp, rbp
    pop rbp
    ret
"""


# msg = """
# .section .mydata, "aw"
# msg:
#     .ascii "Hello world\n"
# """

msg = """
.section .mydata, "aw"
msg:
    .ascii "Hello world\n"
"""

# 这个是Hello world的跳转的地址
meval = """
.section .mddata, "adw"
msg:
    .ascii "eval\n"
"""


def hook_code(uc, address, size, user_data):
    # Disassemble instruction at address
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    code = uc.mem_read(address, size)
    asm = list(md.disasm(code, address))

    print(">>> Tracing instruction at 0x%x, instruction size = 0x%x" %
          (address, size))
    print(">>> Instruction disassembly:")
    for i in asm:
        print("    0x%x:\t%s\t%s" % (i.address, i.mnemonic, i.op_str))

    rsp = uc.reg_read(UC_X86_REG_RSP)
    print(">>> RSP is 0x%x" % rsp)

    rax = uc.reg_read(UC_X86_REG_RAX)
    print(">>> rax is 0x%x" % rax)

    rsi = uc.reg_read(UC_X86_REG_RSI)
    print(">>> rsi is 0x%x" % rsi)

    # Print rsp memory values
    print("RSP Memory:")
    for i in range(rsp, rsp+100, 16):
        data = uc.mem_read(i, 16)
        hex_data = " ".join(f"{b:02x}" for b in data)
        ascii_data = "".join(chr(b) if 32 <= b < 127 else "." for b in data)
        print(f"0x{i:x}: {hex_data}  {ascii_data}")

    # print("RAX Memory:")
    # for i in range(rax, rax+100, 16):
    #     data = uc.mem_read(i, 16)
    #     hex_data = " ".join(f"{b:02x}" for b in data)
    #     ascii_data = "".join(chr(b) if 32 <= b < 127 else "." for b in data)
    #     print(f"0x{i:x}: {hex_data}  {ascii_data}")

    print("MSG Memory:")
    for i in range(0x190000, 0x190000+0x20, 16):
        data = uc.mem_read(i, 16)
        hex_data = " ".join(f"{b:02x}" for b in data)
        ascii_data = "".join(chr(b) if 32 <= b < 127 else "." for b in data)
        print(f"0x{i:x}: {hex_data}  {ascii_data}")

    while True:
        cmd = input("Press enter to step or q to quit: ")
        if cmd == "q":
            exit(0)
        elif cmd == "rax":
            print("RAX Memory:")
            for i in range(rax, rax+100, 16):
                data = uc.mem_read(i, 16)
                hex_data = " ".join(f"{b:02x}" for b in data)
                ascii_data = "".join(chr(b) if 32 <= b <
                                     127 else "." for b in data)
                print(f"0x{i:x}: {hex_data}  {ascii_data}")
        elif cmd == "rsi":
            print("rsi Memory:")
            for i in range(rsi, rsi+100, 16):
                data = uc.mem_read(i, 16)
                hex_data = " ".join(f"{b:02x}" for b in data)
                ascii_data = "".join(chr(b) if 32 <= b <
                                     127 else "." for b in data)
                print(f"0x{i:x}: {hex_data}  {ascii_data}")
        elif cmd == "":
            break


def emulate_program(payload=""):
    # Initialize Unicorn engine
    uc = Uc(UC_ARCH_X86, UC_MODE_64)

    # Map memory for program code and stack
    ADDRESS = 0x100000
    SIZE = 0x10000
    GlobalMsgADDRESS = 0x190000
    uc.mem_map(ADDRESS, SIZE+0x100000)
    uc.mem_map(0x7fffff0000, 0x1000)
    uc.mem_map(0xa646000, 0x1000)

    # Compile assembly code using Keystone
    ks = Ks(KS_ARCH_X86, KS_MODE_64)
    encoding, _ = ks.asm(Main, as_bytes=True)
    msgencoding, _ = ks.asm(msg, as_bytes=True)
    evalencoding, _ = ks.asm(meval, as_bytes=True)
    # Write code to memory and set PC to entry point
    uc.mem_write(ADDRESS, encoding)
    uc.reg_write(UC_X86_REG_RBP, ADDRESS + SIZE - 0x1000 + 8)
    uc.reg_write(UC_X86_REG_RSP, ADDRESS + SIZE - 0x1000)
    uc.reg_write(UC_X86_REG_RIP, ADDRESS)
    uc.mem_write(GlobalMsgADDRESS, msgencoding)
    uc.mem_write(0x0a646c72, evalencoding)

    # Hook read system call
    uc.hook_add(UC_HOOK_CODE, hook_code)
    #uc.hook_add(UC_HOOK_INSN, hook_read, arg=(0, ADDRESS + 0x10, 400))

    # Start emulation

    uc.emu_start(ADDRESS, ADDRESS + len(encoding) + 0x10000)


if __name__ == '__main__':
    emulate_program()