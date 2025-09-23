
    add x1, x2, x3
    sub x4, x5, x6
    sll x7, x8, x9
    slt x10, x11, x12
    sltu x13, x14, x15
    xor x16, x17, x18
    srl x19, x20, x21
    sra x22, x23, x24
    or x25, x26, x27
    and x28, x29, x30

    addi x1, x2, 10
    slli x3, x4, 5
    slti x5, x6, 15
    sltiu x7, x8, 20
    xori x9, x10, 25
    srli x11, x12, 3
    srai x13, x14, 2
    ori x15, x16, 30
    andi x17, x18, 35


    lb x1, 0(x2)
    lh x3, 4(x4)
    lw x5, 8(x6)
    lbu x7, 12(x8)
    lhu x9, 16(x10)
    li x11, 100


    sb x1, 0(x2)
    sh x3, 4(x4)
    sw x5, 8(x6)


    beq x1, x2, label1
    bne x3, x4, label2
    blt x5, x6, label3
    bge x7, x8, label4
    bltu x9, x10, label5
    bgeu x11, x12, label6


    jal x1, label7
    jalr x2, 0(x3)


    lui x1, 0x12345
    auipc x2, 0x67890


    ecall
    ebreak


label1:
    nop
label2:
    nop
label3:
    nop
label4:
    nop
label5:
    nop
label6:
    nop
label7:
    nop