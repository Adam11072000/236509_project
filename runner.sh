#!/bin/bash


usage() {
  echo "Usage: $0 -t [weight|bias|memory] -m [fault_number_models,distribution_bit_flips_models,distribution_faults_models]"
  echo
  echo "Options:"
  echo "  -t  Fault target, choose from weight, bias, or memory."
  echo "  -m  Comma-separated list of fault models, choose from fault_number_models, distribution_bit_flips_models,distribution_faults_models or bits_granularity_models."
  echo " -l target layer name in network"
  echo "  -h  Display this help message."
  exit 1
}

# Initialize variables
FAULT_TARGET=""
FAULT_MODELS=()
LAYER_NAMES=()

# Parse command line flags
while getopts "ht:m:l:" opt; do
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
    l)
      IFS=',' read -ra LAYER_NAMES <<< "$OPTARG"
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
    layer_name=$1
    # fixed bits
    python3 run_injector.py -n 500 -i 15 --target_bits 19 20 21 22 23 --fault_target $FAULT_TARGET --output_dir fault_number_$FAULT_TARGET"_$layer_name" --layer_name $layer_name

    #randomized bits
    python3 run_injector.py -n 500 -i 15 --fault_target $FAULT_TARGET --output_dir fault_number_randomized_bits_$FAULT_TARGET"_$layer_name" --layer_name $layer_name
}

distribution_bit_flips_models(){
    layer_name=$1
    python3 run_injector.py -n 200 -i 25 --fault_target $FAULT_TARGET --output_dir distribution_bit_flips_20_$FAULT_TARGET"_$layer_name" --num_faults 20 --layer_name $layer_name

    python3 run_injector.py -n 200 -i 25 --fault_target $FAULT_TARGET --output_dir distribution_bit_flips_60_$FAULT_TARGET"_$layer_name" --num_faults 60 --layer_name $layer_name

    python3 run_injector.py -n 200 -i 25 --fault_target $FAULT_TARGET --output_dir distribution_bit_flips_100_$FAULT_TARGET"_$layer_name" --num_faults 100 --layer_name $layer_name
}

distribution_faults_models(){
    layer_name=$1 
    python3 run_injector.py -n 200 -i 25 --target_bits 19 20 21 22 23 --fault_target $FAULT_TARGET --output_dir distribution_faults_$FAULT_TARGET"_$layer_name" --num_faults 20 --layer_name $layer_name
}

bits_granularity_models(){
    layer_name=$1 

    python3 run_injector.py -n 100 -i 25 --target_bits 10 11 12 13 14 --fault_target $FAULT_TARGET --output_dir middle_bits_$FAULT_TARGET"_$layer_name" --num_faults 20 --layer_name $layer_name
    python3 run_injector.py -n 100 -i 25 --target_bits 1 2 3 4 5 6 --fault_target $FAULT_TARGET --output_dir low_bits_$FAULT_TARGET"_$layer_name" --num_faults 20 --layer_name $layer_name
    python3 run_injector.py -n 100 -i 25 --target_bits 19 20 21 22 23 --fault_target $FAULT_TARGET --output_dir high_bits_$FAULT_TARGET"_$layer_name" --num_faults 20 --layer_name $layer_name
}


for MODEL in "${FAULT_MODELS[@]}"; do
  if declare -f $MODEL > /dev/null; then
    for layer_name in "${LAYER_NAMES[@]}"; do
      eval "$MODEL $layer_name"
    done
  else
    echo "Invalid fault model: $MODEL. Choose from fault_number_models, distribution_bit_flips_models,distribution_faults_models or bits_granularity_models."
    exit 1
  fi
done
