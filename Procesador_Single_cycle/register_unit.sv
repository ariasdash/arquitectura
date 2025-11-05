module register_unit (
    input  logic         clk,
    input  logic         rst_n,      
    input  logic         RUWr,
    input  logic [4:0]   rs1,
    input  logic [4:0]   rs2,
    input  logic [4:0]   rd,
    input  logic [31:0]  data_in,   //data wr
    output logic [31:0]  data_out1, //RU rs1
    output logic [31:0]  data_out2, //Ru rs2
    output logic [31:0]  regs_debug [31:0]
);
    //se declaran 32 registros de 32 bits
    logic [31:0] regs [31:0];

    assign regs_debug = regs;

    //
    // asegura que no sea el registro x0
    assign data_out1 = (rs1 == 5'd0) ? 32'd0 : regs[rs1];
    assign data_out2 = (rs2 == 5'd0) ? 32'd0 : regs[rs2];

    // Bloque always con reseteo asincrono activo-bajo
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Logica de Reset: Pone todos los registros a 0
            for (int i = 0; i < 32; i = i + 1) begin
                regs[i] <= 32'd0;
            end
        end else begin
            // Logica de operacion normal (con reloj)
            if (RUWr && (rd != 5'd0)) begin
                regs[rd] <= data_in;
            end
        end
    end

endmodule