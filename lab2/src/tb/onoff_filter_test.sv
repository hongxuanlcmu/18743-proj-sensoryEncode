`timescale 1ns / 1ps

module onoff_filter_test;

    reg rst;
    reg filter_center_in;
    reg [0:7] filter_edge_in;
    wire on_center_out;
    wire off_center_out;

    onoff_filter_comp_center DUT (
        .rst(rst),
        .filter_center_in(filter_center_in),
        .filter_edge_in(filter_edge_in),
        .on_center_out(on_center_out),
        .off_center_out(off_center_out) );

    parameter MAX_TIME = 64;
    parameter LEAVEWAY = 5;
    int rand_time[9]; // spike time for the 9 pixels in of the filter

    initial
    begin

        $dumpfile("onoff_filter_comp_center.vcd"); // Change this name as required
        $dumpvars(0, onoff_filter_comp_center);

        // You can insert your own time values (numbers after hash) at your desired input lines

        //// Inputs begin ////
        #1 rst = 0;
        #1 rst = 1;
        #1 rst = 0;
        for (int i = 0; i < 9; i = i + 1) begin
            rand_time[i] = $urandom_range(MAX_TIME);
        end
        for (int i = 0; i < MAX_TIME + LEAVEWAY; i = i + 1) begin
            #1
            for (int j = 0; j < 9; j = j + 1) begin
                if (j < 8) // edge pixels
                    if ((MAX_TIME - rand_time[j]) < i)
                        filter_edge_in[j] = 1'b1;
                    else
                        filter_edge_in[j] = 1'b0;
                else
                    if ((MAX_TIME - rand_time[j]) < i)
                        filter_center_in = 1'b1;
                    else
                        filter_center_in = 1'b0;
            end
        end
        //// Inputs end ////



        #10
        $finish;

    end

endmodule