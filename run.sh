export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


CUDA_VISIBLE_DEVICES=0 python use_dexperts.py \
      --input_file hgissbkh/WMT23-Test \
      --model_path google/gemma-2-2b-it\
      --model_name gemma-2-2b-it \
      --alpha 0.1 \
      --top_p 0.95 \
      --batch_size 2 \
      --temperature 0.0 \
      --prefix_length 10 \
      --longest_n 30 \
      --max_new_tokens 512 \
      --max_length 1000

