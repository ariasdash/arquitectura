addi a1, a1, 5
addi a2, a2, 0
addi a3, a3, 1
addi a4, a4, 0


Loop:
mv t1, a2
mv a2, a3
add a3, a3, t1
sb t1, 0(t2)
addi t2, t2, 4
addi a4, a4, 1
bltu a4, a1, Loop
ebreak