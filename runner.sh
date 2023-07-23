#!/bin/bash


usage() {
  echo "Usage: $0 -t [weight|bias|memory] -m [fault_number_models,distribution_bit_flips_models,distribution_faults_models]"
  echo
  echo "Options:"
  echo "  -t  Fault target, choose from weight, bias, or memory."
  echo "  -m  Comma-separated list of fault models, choose from fault_number_models, distribution_bit_flips_models, or distribution_faults_models."
  echo "  -h  Display this help message."
  exit 1
}

# Initialize variables
FAULT_TARGET=""
FAULT_MODELS=()

# Parse command line flags
while getopts "ht:m:" opt; do
  case ${opt} in
    h)
      usage
      ;;
    t)
      FAULT_TARGET=$OPTARG
      ;;
    m)
      IFS=',' read -ra FAULT_MODELS <<< "$OPTARG"
      ;;
    \?)
      echo "Invalid option -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Verify valid fault_target
if [[ "$FAULT_TARGET" != "weight" && "$FAULT_TARGET" != "bias" && "$FAULT_TARGET" != "memory" ]]; then
  echo "Invalid fault target. Choose from weight, bias, or memory."
  exit 1
fi


fault_number_models(){

    # fixed bits
    python3 run_injector.py -n 500 -i 15 --target_bits 19 20 21 22 23 --fault_target $FAULT_TARGET --output_dir fault_number_$FAULT_TARGET

    #randomized bits
    python3 run_injector.py -n 500 -i 15 --fault_target $FAULT_TARGET --output_dir fault_number_randomized_bits_$FAULT_TARGET    
}

distribution_bit_flips_models(){
    python3 run_injector.py -n 200 -i 25 --fault_target $FAULT_TARGET --output_dir distribution_bit_flips_20_$FAULT_TARGET --num_faults 20

    python3 run_injector.py -n 200 -i 25 --fault_target $FAULT_TARGET --output_dir distribution_bit_flips_60_$FAULT_TARGET --num_faults 60

    python3 run_injector.py -n 200 -i 25 --fault_target $FAULT_TARGET --output_dir distribution_bit_flips_100_$FAULT_TARGET --num_faults 100
}

distribution_faults_models(){
    python3 run_injector.py -n 200 -i 25 --target_bits 19 20 21 22 23 --fault_target $FAULT_TARGET --output_dir distribution_faults_$FAULT_TARGET --num_faults 20
}


for MODEL in "${FAULT_MODELS[@]}"; do
  if declare -f $MODEL > /dev/null; then
    eval "$MODEL"
  else
    echo "Invalid fault model: $MODEL. Choose from fault_number_models, distribution_bit_flips_models, or distribution_faults_models."
    exit 1
  fi
done