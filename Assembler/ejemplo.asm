.data
    number: .word 42
    message: .string "Hello, RISC-V!"

.text
main:
    # Load immediate value
    addi t0, zero, 10
    
    # Arithmetic operations
    add t2, t0, t1
    sub t3, t2, t0
    
    # Branch example
    beq t2, t1, equal
    bne t2, t1, not_equal
    
equal:
    addi a0, zero, 1
    j end
    
not_equal:
    addi a0, zero, 0
    
end:
    # System call
    ecall