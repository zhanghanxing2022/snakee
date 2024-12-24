python "Snake-DQN_lrDouble DQNandCNNNoisyNet.py" \
    --gridsize 15 \
    --num_episodes 1200 \
    --target_update_frequency 5 \
    --lr 5e-5 \
    --num_updates 20 \
    --batch_size 512 \
    --num_games 30 \
    --checkpoint_dir ./CNNnoisynet_Tr5e-5_tgt5_iter20 