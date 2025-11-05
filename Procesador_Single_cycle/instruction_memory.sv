module InstructionMemory (
    // Puerto de instrucciones para el CPU
    input  logic [31:0] addr,
    output logic [31:0] instr,

    // Puerto de lectura para depuración (VGA)
    input  logic [6:0]  debug_addr,  // 7 bits para direccionar 128 entradas
    output logic [31:0] debug_data
);

    // Memoria interna de 128 palabras de 32 bits
    logic [31:0] memory [0:127];

    // Carga inicial de la memoria desde el archivo
    initial begin
        $readmemh("output.bin", memory);
    end

    // Lectura de instrucción (combinacional)
    always_comb begin
        instr = memory[addr[8:2]];
    end

    // Lectura de depuración (combinacional)
    assign debug_data = memory[debug_addr];


endmodule