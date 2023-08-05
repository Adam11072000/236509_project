./runner.sh -t weight -m fault_number_models,distribution_bit_flips_models,distribution_faults_models -l conv1,layer3.0.conv2,layer4.0.conv1,layer4.1.conv2 &
./runner.sh -t memory -m fault_number_models,distribution_bit_flips_models,distribution_faults_models -l conv1,layer3.0.conv2,layer4.0.conv1,layer4.1.conv2 &
./runner.sh -t bias -m fault_number_models,distribution_bit_flips_models,distribution_faults_models -l bn1,layer3.0.bn1,layer3.1.bn1,layer4.1.bn2 &
