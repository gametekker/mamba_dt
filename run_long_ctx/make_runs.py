from run_long_ctx.clean_text import clean_text_to_prompt
from run_long_ctx.clean_text import input_text
import os
for paramcount in ['130m','370m']:
   for ctx_len in [1000,2000,4000]:
      cmd=f"""sudo /usr/local/cuda/bin/ncu --target-processes all --kernel-name selective_scan_fwd_kernel \
   --set full --export {paramcount}_{ctx_len} $(which python3) benchmarks/benchmark_generation_mamba_simple.py \
   --model-name "state-spaces/mamba-{paramcount}" --prompt "{clean_text_to_prompt(input_text,ctx_len)}" \
   --topp 0.9 --temperature 0.7 --repetition-penalty 1.2"""
      os.system(cmd)