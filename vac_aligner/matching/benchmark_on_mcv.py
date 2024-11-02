import json
import argparse
from typing import List

import numpy as np
import pandas as pd
from glob import glob

from .metrics import calculate_wer
from .align import ArmenianAlignerVAC


class Benchmark:
    def __init__(self, target_base: str, predictions_manifest: str):
        """A class performing benchmark for the VAC"""
        self.predictions_manifest = predictions_manifest
        self.matched_texts: List[str] = glob(target_base+"/*.txt")

    def get_benchmark(self):
        columns = ["Text", "ASR Text", "Matched Text", "VAC_WER", "ASR_WER", "Duration", "Path"]
        benchmark = pd.DataFrame(np.zeros([len(self.matched_texts), len(columns)]), columns=columns)

        i = -1
        with open(fr"{self.predictions_manifest}", "r", encoding='utf-8') as f:
            for line in f:
                i += 1
                item = json.loads(line)
                orig_text = item.get("text")
                duration = item["duration"]
                path = item["audio_filepath"]
                if "\\" in path:
                    reference = path.split("\\")[-1].replace(".wav", "_matched.txt")
                else:
                    reference = path.split("/")[-1].replace(".wav", "_matched.txt")
                matched_text_file = [x for x in self.matched_texts if reference in x]
                assert len(matched_text_file) == 1, f"matched_text_file: {matched_text_file}"
                matched_text_file = matched_text_file[0]
                with open(matched_text_file, "r", encoding='utf-8') as f:
                    matched_text = f.read().strip()

                pred_text = item['pred_text']
                asr_wer = calculate_wer(orig_text,pred_text)
                vac_wer = calculate_wer(orig_text, matched_text)
                benchmark.iloc[i, :] = [orig_text, pred_text, matched_text, vac_wer, item.get('wer', asr_wer), duration, path]
        benchmark.drop_duplicates(inplace=True)   # if somehow some rows are empty: 0s
        return benchmark

    @staticmethod
    def analyze_and_save_benchmark(benchmark: pd.DataFrame, output_file: str) -> pd.DataFrame:
        benchmark.VAC_WER = benchmark.VAC_WER.apply(lambda x: 0. if x == 0 else round(x, 3))
        benchmark.ASR_WER = benchmark.ASR_WER.apply(lambda x: 0. if x == 0 else round(x, 3))
        print(f"Mean WER: {round(100 * benchmark.VAC_WER.sum() / len(benchmark), 3)}%")
        print(f"Accuracy (ASR): {round(100 * len(benchmark[benchmark.ASR_WER == 0]) / len(benchmark.ASR_WER), 3)}%")
        print(f"Accuracy (VAC): {round(100 * len(benchmark[benchmark.VAC_WER == 0]) / len(benchmark.VAC_WER), 3)}%")
        benchmark.to_csv(output_file.replace(".txt", ".csv"), index=False)
        print(benchmark.sample(10))
        return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the predictions manifest and output a combined transcript.")
    parser.add_argument("--output_file", type=str, help="Path where to save the combined MCV transcript.", default=r"D:\CapstoneThesisArmASR\publish\experiments\combined.txt")
    parser.add_argument("--target_base", type=str, help="Path where need to save matched texts", default=r"D:\CapstoneThesisArmASR\publish\experiments\matches")
    parser.add_argument("--predictions_manifest", type=str, help="Path to the manifest file storing predictions.",
                        default=r"C:\Users\GIGABYTE\workspace\Nvidia\NeMo\test_with_predictions17_replaced.json")
    parser.add_argument("--save_manifest_path", type=str, help="Path to save matching related manifest", default="D:\CapstoneThesisArmASR\publish\experiments\matches.json")
    args = parser.parse_args()

    chunks, combined_transcript = ArmenianAlignerVAC.combine_transcript(args.predictions_manifest,
                                                                        args.output_file, ending_punctuations="․,։")
    print("Combined the MCV files into single transcript")
    # AlignerVAC.align(combined_transcript, chunks, args.output_file, target_base=args.target_base)
    matches_sorted = ArmenianAlignerVAC(combined_transcript, chunks, args.save_manifest_path, target_base=args.target_base).align(0.35)

    """ Logs You should get (timestamp can vary from system to system)
    1it [00:00,  1.90it/s]~/AppData/Local/Temp/ipykernel_33732/804255406.py:31: RuntimeWarning: divide by zero encountered in divide
      cer = d[n][m] / float(n)
    48it [00:17,  2.64it/s]['Ինչպես հաղորդվում է՝ ֆիլմը կատակերգությու', 'Ինչպես հավաք է, մկատագիտությունը, ։', 0.5405405405405406, 3152, 3193, 46]
    49it [00:20,  1.20s/it]['հաղորդվում է՝ ֆիլմը կատակերգություն է լինելու։ Ռուսաստանում այն ստացավ պլատինե', 'Սոեբում նաի ստացավ լա, եստտավե ան համ, ավաստան։', 0.7391304347826086, 3159, 3237, 47]
    137it [00:45,  6.89it/s][' Է՛հ, գնա՛', ' է՛՜հ, գնա է։', 0.5, 8152, 8162, 136]
    609it [02:55,  2.21it/s]['րավորությունները։ Մարդիկ կրում են կոնտակտային ոսպնյակներ բազմաթիվ', 'Մարդիկ կրում են կոնտակտային ոսպնյակներ բազմաթիվ պատճառներով։', 0.4915254237288136, 36695, 36760, 607]
    1108it [05:28,  4.85it/s][' հա՜, հա՜, հա՜…։Այն', 'Հա՛՜, հա՜, հա..', 0.5, 67528, 67547, 1107]
    1477it [07:12,  7.10it/s][' - Ո՞ւր ես գնում։', 'Ես արնր ես գնում։', 0.38461538461538464, 89741, 89758, 1477]
    1818it [08:52,  3.42it/s][' գիշեր։- նրա մեջ մ', '-Բարի լիշեր.', 0.8571428571428571, 110543, 110561, 1818]
    2632it [12:36,  2.28it/s][' էլի կասե՞ս սուտ', ' Էլիկ ասես Սուտա.', 0.38461538461538464, 159899, 159915, 2631]
    2935it [13:57,  5.08it/s][' Տգետ, պառավ, գինեվաճառ…։Նրա', 'Տկետ բառավ կին է վաճառը։', 0.4, 178049, 178077, 2934]
    3789it [17:57,  1.56it/s]['նկարները և այլն։ Ապրել են գաղութներով՝ տաք ծովերում և ոչ մեծ խորութ', 'Ապրել են գաղութներով, տաք ծովերում և ոչ մեծ խորություններում։', 0.4642857142857143, 229251, 229318, 3787]
    4281it [20:14,  3.53it/s]
    """
    print("Finished Matching the texts")
    benchmark = Benchmark(args.target_base, args.predictions_manifest)
    stats = benchmark.get_benchmark()
    benchmark.analyze_and_save_benchmark(stats, args.output_file)
