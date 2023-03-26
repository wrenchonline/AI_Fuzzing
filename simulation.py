from unicorn import *
from unicorn.x86_const import *
from keystone import *
import capstone
import json

from queue import Queue

debug_mode = True
# 创建一个空队列
q = Queue()


#fuzzing + ai

Main = """
_start:
    push rbp
    sub rsp, 20
    call overflow
    add rsp, 20
    xor rax, 0
    ret
overflow:
    push rbp
    mov rbp, rsp
    sub rsp, 12
    mov rcx ,100
    mov rdi,0x190000
    cld
    repne scasb
    sub rdi, 0x190000
    mov rax,rdi
    mov rsi, 0x190000
    lea rdi, [rsp+8]
    MOV rcx, rax
    rep movsb
    add rsp, 12
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
    .ascii "%s"
"""

# 这个是Hello world的跳转的地址
meval = """
.section .mddata, "adw"
msg:
    .ascii "eval\n"
"""

opt = list()

#least_RSP = None


def hook_code(uc, address, size, user_data):
    global opt
    mdisassembly = None
    isCall = False
    isRet = False
    #global least_RSP
    assert user_data, f"Invalid action {user_data}"
    q = user_data["queue"]
    # q.put("111")
    # Disassemble instruction at address
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    code = uc.mem_read(address, size)
    asm = list(md.disasm(code, address))
    if debug_mode:
        print(">>> Tracing instruction at 0x%x, instruction size = 0x%x" %
              (address, size))
        print(">>> Instruction disassembly:")
    for i in asm:
        if debug_mode:
            print("    0x%x:\t%s\t%s" % (i.address, i.mnemonic, i.op_str))
        disassembly = {
            'address': i.address,
            'mnemonic': i.mnemonic,
            'op_str': i.op_str,
        }
        # 是否有call
        if i.mnemonic == "call":
            isCall = True
        # 是否有ret
        if i.mnemonic == "ret":
            isRet = True
            # 读取返回地址
            # least_RSP = uc.mem_read(rsp, 8)
            # print(hex_data)
        mdisassembly = disassembly
        opt.append(disassembly)

    rsp = uc.reg_read(UC_X86_REG_RSP)
    if debug_mode:
        print(">>> RSP is 0x%x" % rsp)

    rax = uc.reg_read(UC_X86_REG_RAX)
    if debug_mode:
        print(">>> rax is 0x%x" % rax)

    rsi = uc.reg_read(UC_X86_REG_RSI)
    if debug_mode:
        print(">>> rsi is 0x%x" % rsi)

    rbp = uc.reg_read(UC_X86_REG_RBP)
    if debug_mode:
        print(">>> rbp is 0x%x" % rbp)
    rdi = uc.reg_read(UC_X86_REG_RDI)
    if debug_mode:
        print(">>> rdi is 0x%x" % rdi)

    # Print rsp memory values
    if debug_mode:
        print("RSP Memory:")
    for i in range(rsp, rsp+100, 16):
        data = uc.mem_read(i, 16)
        hex_data = " ".join(f"{b:02x}" for b in data)
        ascii_data = "".join(chr(b) if 32 <= b < 127 else "." for b in data)
        if debug_mode:
            print(f"0x{i:x}: {hex_data}  {ascii_data}")

    # 读取返回地址
    hex_data = uc.mem_read(rsp, 8)
    if debug_mode:
        print(hex_data)
    # if least_RSP == hex_data:
    #     pass
    # global isCall
    debug_msg = {"disassembly": mdisassembly,
                 "isCall": isCall, "return_adress": hex_data, "isRet": isRet}
    q.put(debug_msg)
    # print("sssss")

    # print("MSG Memory:")
    # for i in range(0x190000, 0x190000+0x20, 16):
    #     data = uc.mem_read(i, 16)
    #     hex_data = " ".join(f"{b:02x}" for b in data)
    #     ascii_data = "".join(chr(b) if 32 <= b < 127 else "." for b in data)
    #     print(f"0x{i:x}: {hex_data}  {ascii_data}")

    # while True:
    #     cmd = input("Press enter to step or q to quit: ")
    #     if cmd == "q":
    #         with open('bin_array.json', 'w+') as f:
    #             json.dump(opt, f)
    #         exit(0)
    #     elif cmd == "rax":
    #         print("RAX Memory:")
    #         for i in range(rax, rax+100, 16):
    #             data = uc.mem_read(i, 16)
    #             hex_data = " ".join(f"{b:02x}" for b in data)
    #             ascii_data = "".join(chr(b) if 32 <= b <
    #                                  127 else "." for b in data)
    #             print(f"0x{i:x}: {hex_data}  {ascii_data}")
    #     elif cmd == "rsi":
    #         print("rsi Memory:")
    #         for i in range(rsi, rsi+100, 16):
    #             data = uc.mem_read(i, 16)
    #             hex_data = " ".join(f"{b:02x}" for b in data)
    #             ascii_data = "".join(chr(b) if 32 <= b <
    #                                  127 else "." for b in data)
    #             print(f"0x{i:x}: {hex_data}  {ascii_data}")
    #     elif cmd == "rbp":
    #         print("rbp Memory:")
    #         for i in range(rbp, rbp+100, 16):
    #             data = uc.mem_read(i, 16)
    #             hex_data = " ".join(f"{b:02x}" for b in data)
    #             ascii_data = "".join(chr(b) if 32 <= b <
    #                                  127 else "." for b in data)
    #             print(f"0x{i:x}: {hex_data}  {ascii_data}")
    #     elif cmd == "":
    #         break


def emulate_program(queue, payload):
    # Initialize Unicorn engine
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    global msg

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
    new_msg = msg % payload
    msgencoding, _ = ks.asm(new_msg, as_bytes=True)
    evalencoding, _ = ks.asm(meval, as_bytes=True)
    # Write code to memory and set PC to entry point
    uc.mem_write(ADDRESS, encoding)
    uc.reg_write(UC_X86_REG_RBP, ADDRESS + SIZE - 0x1000)
    uc.reg_write(UC_X86_REG_RSP, ADDRESS + SIZE - 0x1000)
    uc.reg_write(UC_X86_REG_RIP, ADDRESS)
    uc.mem_write(GlobalMsgADDRESS, msgencoding)
    uc.mem_write(0x0a646c72, evalencoding)

    # Hook read system call
    uc.hook_add(UC_HOOK_CODE, hook_code, user_data={"queue": queue})
    #uc.hook_add(UC_HOOK_INSN, hook_read, arg=(0, ADDRESS + 0x10, 400))

    # Start emulation
    try:
        uc.emu_start(ADDRESS, ADDRESS + len(encoding) + 0x10000)
    except Exception as e:
        print("Clash\n")
    finally:
        queue.put(None)


# if __name__ == '__main__':
#     emulate_program(q, '11111')
