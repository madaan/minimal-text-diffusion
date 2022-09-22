import os, sys, glob

full_lst = glob.glob(sys.argv[1])
top_p = -1.0 if len(sys.argv) < 2 else sys.argv[2]
print(f'top_p = {top_p}')
pattern_ = 'model' if len(sys.argv) < 3 else sys.argv[3]
print(f'pattern_ = {pattern_}', sys.argv[3])
# print(full_lst)

output_lst = []
for lst in full_lst:
    print(lst)
    try:
        tgt = sorted(glob.glob(f"{lst}/{pattern_}*pt"))[-1]
        lst = os.path.split(lst)[1]
        print("lst = ", lst)
        num = 1
    except Exception as e:
        raise e
        continue
    
    print(f"Using {tgt} for generating samples")
    # model_arch_ = lst.split('_')[5-num]
    model_arch = 'transformer'
    mode =  'text' #or '1d-unet' in model_arch_

    dim_ = 16


    if 'synth32' in lst:
        kk = 32
    elif 'synth128' in lst:
        kk = 128

    noise_schedule = 'linear'
    dim = 76
    num_channels = 16

    out_dir = 'generation_outputs'
    num_samples = 50


    COMMAND = f'python src/train_infer/{mode}_sample.py ' \
    f'--model_path {tgt} --batch_size 50 --num_samples {num_samples} --top_p {top_p} ' \
    f'--out_dir {out_dir} '
    print("Running decode")
    print(COMMAND)
    os.system(COMMAND)


    # COMMAND = f"python scripts/ppl_under_ar.py " \
    #           f"--model_path {tgt} " \
    #           f"--modality {modality}  --experiment random " \
    #           f"--model_name_or_path {model_name_path} " \
    #           f"--input_text {out_path2}  --mode eval"

print('output lists:')
print("\n".join(output_lst))