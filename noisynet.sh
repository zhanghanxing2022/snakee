python "Snake-DQN_lrDouble DQNandNoisyNet.py"\
    --gridsize 15 \
    --num_episodes 1200 \
    --target_update_frequency 5 \
    --lr 1e-3 \
    --num_updates 20 \
    --batch_size 512 \
    --num_games 30 \
    --checkpoint_dir ./noisynet_1.4_Tr1e-3_tgt5_iter20