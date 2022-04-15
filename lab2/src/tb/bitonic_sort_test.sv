// Testbench for a 32-input bitonic sorter

// Input lines make 1->0 transitions at different times coming in
// at the 32 input lines in an unsorted manner. You can play around with it
// as you like.
// At the output of the sorter, the lines should have 1->0 transitions in
// ascending order.

// You DON'T need to modify this file to verify 32-bit sorter. But for testing
// sorters with other sizes, you can follow the comments below.

`timescale 1ns / 1ps

module bitonic_sort_test;

    reg [0:31] raw_in; // Change this value to represent your input size
                       // For example, if you want to test a 4-input sorter, it should be [0:3]
    wire [0:31] sorted_out; // As above

    bitonic_sort_32 DUT (.sorted_out(sorted_out), .raw_in(raw_in)); // Change this instantiation
                                                                    // as per the sorter you wish to test

    // reg [0:3] raw_in; // Change this value to represent your input size
    //                    // For example, if you want to test a 4-input sorter, it should be [0:3]
    // wire [0:3] sorted_out; // As above

    // bitonic_sort_32 #(2, 4) DUT (.sorted_out(sorted_out), .raw_in(raw_in)); // Change this instantiation
    //                                                                 // as per the sorter you wish to test
    parameter MAX_INPUT = 64;
    parameter LEAVEWAY = 5;
    int inputs[32];

    initial
    begin

        $dumpfile("bitonic_sort_32.vcd"); // Change this name as required
        $dumpvars(0, bitonic_sort_test);

        // You can insert your own time values (numbers after hash) at your desired input lines

        //// Inputs begin ////
        for (int i = 0; i < 32; i = i + 1) begin
            inputs[i] = $urandom_range(MAX_INPUT);
        end
        for (int i = 0; i < MAX_INPUT + LEAVEWAY; i = i + 1) begin
            #1
            for (int j = 0; j < 32; j = j + 1) begin
                if ((MAX_INPUT - inputs[j]) < i)
                    raw_in[j] = 1'b1;
                else
                    raw_in[j] = 1'b0;
            end
        end
        //// Inputs end ////



        #10
        $finish;

    end

endmodule
