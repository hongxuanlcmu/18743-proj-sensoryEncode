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

    initial
    begin

        $dumpfile("bitonic_sort_32.vcd"); // Change this name as required
        $dumpvars(0, bitonic_sort_test);

        // You can insert your own time values (numbers after hash) at your desired input lines

        //// Inputs begin ////

        
        
        //// Inputs end ////



        #200
        $finish;

    end

endmodule
