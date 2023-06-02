source venv/bin/activate

# Twitter Slow Forgetting BERT Mini Model

# ORIS Delta = 8
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "rl" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "89,8,9,60,64" \
 --save_prefix "_c9394_f25_sgmd_slow_tw" --rl_save_ckpt "9394"

# Uncertainty
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "unc" \
 --unc_decay --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "89,8,9,60,64" \
 --save_prefix "_c9394_f25_sgmd_slow_tw" --rl_save_ckpt "9394"

# Random
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "random" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "89,8,9,60,64" \
 --save_prefix "_c9394_f25_sgmd_slow_tw" --rl_save_ckpt "9394"

# Diversity
python run_diversity.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "diversity" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "89" \
 --save_prefix "_c9394_f25_sgmd_slow_tw"

# Reddit Slow Forgetting BERT Mini Model

# ORIS Delta = 8
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "rl" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "10,1,46,76,60" \
 --save_prefix "_c9394_f25_sgmd_slow_rd" --rl_save_ckpt "9394"

# Uncertainty
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "unc" \
 --unc_decay --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "10,1,46,76,60" \
 --save_prefix "_c9394_f25_sgmd_slow_rd" --rl_save_ckpt "9394"

# Random
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "random" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "10,1,46,76,60" \
 --save_prefix "_c9394_f25_sgmd_slow_tw" --rl_save_ckpt "9394"

# Diversity
python run_diversity.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "diversity" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "10" \
 --save_prefix "_c9394_f25_sgmd_slow_tw"
# Twitter Fast Forgetting BERT Mini Model

# ORIS Delta = 8
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "rl" \
 --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "60,11,46,58,89" \
 --save_prefix "_c9394_f25_expn_fast_tw" --rl_save_ckpt "9394"

# Uncertainty
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "unc" \
 --unc_decay --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "60,11,46,58,89" \
 --save_prefix "_c9394_f25_expn_fast_tw" --rl_save_ckpt "9394"

# Random
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "random" \
 --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "60,11,46,58,89" \
 --save_prefix "_c9394_f25_expn_fast_tw" --rl_save_ckpt "9394"

# Diversity
python run_diversity.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "diversity" \
 --unc_decay --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "60" \
 --save_prefix "_c9394_f25_expn_fast_tw"

# Reddit Fast Forgetting BERT Mini Model

# ORIS Delta = 8
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "rl" \
 --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "666,10,15,60,76" \
 --save_prefix "_c9394_f25_expn_fast_rd" --rl_save_ckpt "9394"

# Uncertainty
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "unc" \
 --unc_decay --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "666,10,15,60,76" \
 --save_prefix "_c9394_f25_expn_fast_rd" --rl_save_ckpt "9394"

# Random
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "random" \
 --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "666,10,15,60,76" \
 --save_prefix "_c9394_f25_expn_fast_rd" --rl_save_ckpt "9394"

# Diversity
python run_diversity.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "prajjwal1/bert-mini" --al_sampling_type "diversity" \
 --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "666" \
 --save_prefix "_c9394_f25_expn_fast_rd"

# Twitter Slow Forgetting mBERT Base Model

# ORIS Delta = 8
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "rl" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "80,54,60,75,89" \
 --save_prefix "_mbert_c9394_f25_sgmd_slow_tw" --rl_save_ckpt "9394"

# Uncertainty
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "unc" \
 --unc_decay --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "80,54,60,75,89" \
 --save_prefix "_mbert_c9394_f25_sgmd_slow_tw" --rl_save_ckpt "9394"

# Random
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "random" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "80,54,60,75,89" \
 --save_prefix "_mbert_c9394_f25_sgmd_slow_tw" --rl_save_ckpt "9394"

# Diversity
python run_diversity.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "diversity" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "80" \
 --save_prefix "_mbert_c9394_f25_sgmd_slow_tw"

# Reddit Slow Forgetting mBERT Base Model

# ORIS Delta = 8
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "rl" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "76,3,15,98,99" \
 --save_prefix "_mbert_c9394_f25_sgmd_slow_rd" --rl_save_ckpt "9394"

# Uncertainty
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "unc" \
 --unc_decay --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "76,3,15,98,99" \
 --save_prefix "_mbert_c9394_f25_sgmd_slow_rd" --rl_save_ckpt "9394"

# Random
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "random" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "76,3,15,98,99" \
 --save_prefix "_mbert_c9394_f25_sgmd_slow_tw" --rl_save_ckpt "9394"

# Diversity
python run_diversity.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "diversity" \
 --mem_decay "0.3,9,1" --mem_decay_type "sigmoid" --seed_values "76" \
 --save_prefix "_mbert_c9394_f25_sgmd_slow_tw"

# Twitter Fast Forgetting mBERT Base Model

# ORIS Delta = 8
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "rl" \
 --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "80,7,18,64,73" \
 --save_prefix "_mbert_c9394_f25_expn_fast_tw" --rl_save_ckpt "9394"

# Uncertainty
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "unc" \
 --unc_decay --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "80,7,18,64,73" \
 --save_prefix "_mbert_c9394_f25_expn_fast_tw" --rl_save_ckpt "9394"

# Random
python run_oal_oris.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "random" \
 --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "80,7,18,64,73" \
 --save_prefix "_mbert_c9394_f25_expn_fast_tw" --rl_save_ckpt "9394"

# Diversity
python run_diversity.py \
 --data_path "data/emotion_twitter" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "diversity" \
 --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "80" \
 --save_prefix "_mbert_c9394_f25_expn_fast_tw"

# Reddit Fast Forgetting mBERT Base Model

# ORIS Delta = 8
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "rl" \
 --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "53,10,15,98,666" \
 --save_prefix "_mbert_c9394_f25_expn_fast_rd" --rl_save_ckpt "9394"

# Uncertainty
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "unc" \
 --unc_decay --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "53,10,15,98,666" \
 --save_prefix "_mbert_c9394_f25_expn_fast_rd" --rl_save_ckpt "9394"

# Random
python run_oal_oris.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "random" \
 --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "53,10,15,98,666" \
 --save_prefix "_mbert_c9394_f25_expn_fast_rd" --rl_save_ckpt "9394"

# Diversity
python run_diversity.py \
 --data_path "data/emotion_reddit" --al_bert_model_name "bert-base-multilingual-uncased" --al_sampling_type "diversity" \
 --mem_decay "0.6,-19,1" --mem_decay_type "exponential" --seed_values "53" \
 --save_prefix "_mbert_c9394_f25_expn_fast_rd"
