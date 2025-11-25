module pc (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [31:0] next_pc,
    
    output logic [31:0] address
);

    // Registro del PC
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            address <= 32'h0000_0000;
        end else begin
            // Simplemente cargamos lo que el Top Level calculó
            address <= next_pc;
        end
    end

endmodule