#!/bin/bash


input=$1


runner_generic(){
    target=$1

    # fault in relation to bits
    python3 tester_conv.py -n 100 -i 25 --target_bits 10 11 12 13 14 --fault_target $target --output_dir middle_bits_$target --num_faults 20
    python3 tester_conv.py -n 100 -i 25 --target_bits 1 2 3 4 5 6 --fault_target $target --output_dir low_bits_$target --num_faults 20
    python3 tester_conv.py -n 100 -i 25 --target_bits 19 20 21 22 23 --fault_target $target --output_dir high_bits_$target --num_faults 20

    # fault in relation to fault number, take fixed bits
    python3 tester_conv.py -n 500 -i 15 --target_bits 19 20 21 22 23 --fault_target $target --output_dir fault_number_$target

    # fault in relation to fault number, take randomized bits bits
    python3 tester_conv.py -n 500 -i 15 --fault_target $target --output_dir fault_number_randomized_bits_$target

    # in relation to distribution of faults, will contain gaussian and uniform
    python3 tester_conv.py -n 200 -i 25 --target_bits 19 20 21 22 23 --fault_target $target --output_dir distribution_faults_$target --num_faults 20

    # in relation to distribution of bit flips, will contain gaussian and uniform
    python3 tester_conv.py -n 200 -i 25 --fault_target $target --output_dir distribution_bit_flips_$target --num_faults 20
}


runner_generic $input
