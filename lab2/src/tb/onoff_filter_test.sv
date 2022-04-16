`timescale 1ns / 1ps

module onoff_filter_test;

    reg filter_center_in;
    reg [0:7] filter_edge_in;
    wire on_center_out;
    wire off_center_out;

    onoff_filter_comp_center DUT (
        .filter_center_in(filter_center_in),
        .filter_edge_in(filter_edge_in),
        .on_center_out(on_center_out),
        .off_center_out(off_center_out) );

    parameter MAX_TIME = 64;
    parameter LEAVEWAY = 5;
    int rand_time[9]; // spike time for the 9 pixels in of the filter
    int _;
    logic crtTestOnCenter;
    logic crtTestOffCenter;
    int sorted_edges[8];
    logic expectOnCenter;
    logic expectOffCenter;
    int expectOnCenterTime;
    int expectOffCenterTime;
    int seed;
    function printRandTime();
        for (int i = 0; i < 9; i = i + 1) begin
            $display("rand_time[%d] = %d", i, rand_time[i]);
        end        
    endfunction

    initial
    begin

        $dumpfile("onoff_filter_comp_center.vcd"); // Change this name as required
        $dumpvars(0, onoff_filter_comp_center);

        // You can insert your own time values (numbers after hash) at your desired input lines

        //// Inputs begin ////
        // seed random
        seed = $system("exit $(($RANDOM % 255))");
        _ = $urandom(seed);
        $display("Using rand seed: %d", seed);
        for (int test = 0; test < 10000; test = test + 1) begin
            $display("Test number: %d", test);
            crtTestOnCenter = 1'b0;
            crtTestOffCenter = 1'b0;
            filter_edge_in = 0;
            filter_center_in = 0;
            #1
            for (int i = 0; i < 9; i = i + 1) begin
                rand_time[i] = $urandom_range(MAX_TIME);
                // $display("rand_time[%d] = %d", i, rand_time[i]);
                if (i < 8)
                    sorted_edges[i] = rand_time[i];
            end
            // Generate expected output and output time
            sorted_edges.rsort(); // edge pixel spike times sorted in decensending order
            expectOnCenter = 1'b0;
            expectOffCenter = 1'b0;
            if (rand_time[8] <= sorted_edges[4]) begin // onCenter
                expectOnCenter = 1'b1;
                expectOnCenterTime = rand_time[8];
            end
            if (rand_time[8] >= sorted_edges[3]) begin // offCenter
                expectOffCenter = 1'b1;
                expectOffCenterTime = sorted_edges[3];
            end

            #1
            for (int i = 0; i < MAX_TIME + LEAVEWAY; i = i + 1) begin
                for (int j = 0; j < 9; j = j + 1) begin
                    if (j < 8) // edge pixels
                        if (i < rand_time[j])
                            filter_edge_in[j] = 1'b0;
                        else
                            filter_edge_in[j] = 1'b1;
                    else // center pixel
                        if (i < rand_time[j])
                            filter_center_in = 1'b0;
                        else
                            filter_center_in = 1'b1;
                end
                #1
                if (on_center_out == 1'b1 && !crtTestOnCenter) begin
                    crtTestOnCenter = 1'b1;
                    $display("time:%d, on center spike", i);
                    // Test if this spike is as expected
                    if (!expectOnCenter) begin
                        printRandTime();
                        $fatal("ERROR: Should not be on center!\n");
                    end
                    else if (expectOnCenterTime != i) begin
                        printRandTime();
                        $fatal("ERROR: On center expected %d!\n", expectOnCenterTime);
                    end
                end
                if (off_center_out == 1'b1 && !crtTestOffCenter) begin
                    crtTestOffCenter = 1'b1;
                    $display("time:%d, off center spike", i);
                    // Test if this spike is as expected
                    if (!expectOffCenter) begin
                        printRandTime();
                        $fatal("ERROR: Should not be off center!\n");
                    end
                    else if (expectOffCenterTime != i) begin
                        printRandTime();
                        $fatal("ERROR: Off center expected %d!\n", expectOffCenterTime);
                    end
                end
            end
        end
        //// Inputs end ////



        #10
        $finish;

    end

endmodule
