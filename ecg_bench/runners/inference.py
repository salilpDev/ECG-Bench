import numpy as np
import torch
from tqdm import tqdm

def compare_after_sequence(s1, s2, sequence):
    # Find where the sequence appears in each string
    i1 = s1.find(sequence)
    i2 = s2.find(sequence)

    # If the sequence isn't found in either string, stop early
    if i1 == -1 or i2 == -1:
        print("Sequence not found in one or both strings.")
        return None

    # Slice both strings starting from the found sequence
    s1_sub = s1[i1:]
    s2_sub = s2[i2:]

    # Compare character by character
    min_len = min(len(s1_sub), len(s2_sub))
    matches = sum(c1 == c2 for c1, c2 in zip(s1_sub[:min_len], s2_sub[:min_len]))
    accuracy = matches / max(len(s1_sub), len(s2_sub)) * 100
    
    return accuracy



def tester_chat(model, dataloader, tokenizer, args, train_utils):
    model.eval()
    len_of_batch = 0
    dev_count = 0
    gt_answers = []
    gen_answers = []
    acc_dic = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Testing {args.model}", position=0, leave=True)):
            if batch is None:
                print("Skipping invalid batch")
                continue

            try:
                
                gt_input_ids = batch["input_ids"]
                gt_attention_mask = batch["attn_mask"]
                gt_signal = batch["gt_signal"]
                # Generate full model output for the given input
                out = model.generate_chat(
                    input_ids=gt_input_ids,
                    attention_mask=gt_attention_mask, 
                    tokenizer=tokenizer,
                )
            
                decoded_out = tokenizer.batch_decode(out[-512:], skip_special_tokens=False)[0]

                gt_out = tokenizer.batch_decode(gt_input_ids, skip_special_tokens=False)[0]
                gt_signal_out = tokenizer.batch_decode(gt_signal, skip_special_tokens=False)[0]
                
                print(len(decoded_out))
                print(len(gt_signal_out))


                
                
                gt_answers.append(gt_out)
                gen_answers.append(decoded_out)
                
            except Exception as e:
                print("\nError occurred during evaluation:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {e!s}")
                print(f"Error location: {e.__traceback__.tb_lineno}")
                gt_answers.append("")
                gen_answers.append("")

            len_of_batch += 1

            if args.dev:
                print(f"\nCompleted batch {batch_idx}. Total conversations processed: {len_of_batch}")
                dev_count += 1
                if dev_count == 25:
                    print("\nDev mode: Stopping after 25 batches")
                    break

    print("\nCalculating metrics...")

    try:
        all_metrics = train_utils.evaluate_strings(gt_answers, gen_answers, args.device)
        
        print("\nMetrics calculated successfully:")
        print(f"BLEU: {all_metrics['BLEU']}")
        print(f"METEOR: {all_metrics['METEOR']}")
        print(f"ROUGE-L: {all_metrics['ROUGE']['rouge-l']}")
        print(f"BERTScore F1: {np.mean(all_metrics['BERTSCORE']['hf-f1'])}")
        print(f"Accuracy: {sum(acc_dic)/1}")
    except Exception as e:
        print("\nError during metric calculation:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e!s}")
        all_metrics = {
            "BLEU": 0,
            "METEOR": 0.0,
            "ROUGE": {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0},
            "BERTSCORE": {"hf-prec": [0.0], "hf-rec": [0.0], "hf-f1": [0.0]},
            "ACC": 0.0,
        }

    print("\nEvaluation complete!")
    return {
        "metrics": all_metrics,
        "qa_results": {
            "gt_answers": gt_answers,
            "gen_answers": gen_answers,
        },
    }
