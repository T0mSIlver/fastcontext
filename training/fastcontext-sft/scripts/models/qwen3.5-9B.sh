MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16   # num_attention_heads
   --num-query-groups 4       # num_key_value_heads
   --kv-channels 256          # head_dim
   --num-layers 32            # num_hidden_layers
   --hidden-size 4096         # hidden_size
   --ffn-hidden-size 12288    # intermediate_size
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25     # partial_rotary_factor
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # qwen3.5 specific
   --attention-output-gate
)