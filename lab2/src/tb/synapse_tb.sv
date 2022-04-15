// Testbench for FSM synapse

// Verify your design by inserting input stimuli 
// Check for correct state transistions, and verify if the output readout is correct
// Please make sure - 
//  1. input spike stimuli should be 8 cycle wide
//  2. input spike should be passed one clock cycle after gclk
//  3. within the 24-cycle wide gclk, the last input spike should be within the 16th cycle


`timescale 1ns / 1ps

module synapse_tb;

    reg input_spike, inc, dec, aclk, gclk, rst;
    wire out;
    wire [0:2] weight;

    integer i;
    
    synapse DUT (.out(out), .weight(weight), .input_spike(input_spike), .inc(inc), .dec(dec), .aclk(aclk), .gclk(gclk), .rst(rst));
    
    initial
    begin

        $dumpfile("synapse.vcd");
        $dumpvars(0, synapse_tb);

        // You can insert your own time values (numbers after hash) at your desired input lines

        //// Inputs begin ////

        input_spike = 0;
        inc = 0;
        dec = 0;
        aclk = 1;
        gclk = 0;
        rst = 1;

        #25
        rst = 0;

        @(posedge gclk);
        
        #5
        input_spike = 1;
        
        #8
        input_spike = 0;
        
        #2
        inc = 1;
        
        @(posedge gclk);

        #1
        input_spike = 1;

        #8
        input_spike = 0;

        @(posedge gclk);

        #3
        input_spike = 1;

        #8
        input_spike = 0;

        #2
        inc = 0;
        dec = 1;

        @(posedge gclk);
        #4
        input_spike = 1;

        #8
        input_spike = 0;

        #3
        inc = 1;
        dec = 0;

        @(posedge gclk);

        #2
        input_spike = 1;

        #8
        input_spike = 0;
        
        @(posedge gclk);

        #6
        input_spike = 1;

        #8
        input_spike = 0;
        
        @(posedge gclk);

        #5
        input_spike = 1;

        #8
        input_spike = 0;
        
        @(posedge gclk);

        #7
        input_spike = 1;

        #8
        input_spike = 0;

        @ (posedge gclk);
        #5
        rst = 1;

        #5
        rst = 0;
        inc = 0;
        dec = 0;

        //// Inputs end ////

        #200
        $finish;

    end
    
    always
    #0.5 aclk = !aclk;

    initial i = 0;
    always @ (posedge aclk)
    begin
        i = i % 24;
        if (i==0) gclk = 1;
        if (i==1) gclk = 0;
        i = i + 1;
    end

endmodule
