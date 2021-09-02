import re
import os
import subprocess

class BLEUEvaluator(object):
    """Evaluator calling multi-bleu.perl."""

    def _get_bleu_script(self):
        return "multi-bleu.perl"

    def name(self):
        return "BLEU"

    def score(self, labels_files, predictions_path):
        bleu_script = self._get_bleu_script()
        try:
            third_party_dir = get_script_dir()
        except RuntimeError as e:
            print("%s", str(e))
            return None

        if not isinstance(labels_files, list):
            labels_files = [labels_files]

        try:
            cmd = 'perl %s %s < %s' % (os.path.join(third_party_dir, bleu_script),
                                       " ".join(labels_files),
                                       predictions_path)
            bleu_out = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                shell=True)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            return float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                msg = error.output.strip()
                print(
                    "{} script returned non-zero exit code: {}".format(bleu_script, msg))
            return None


def get_script_dir():
    utils_dir = os.path.dirname(__file__)
    script_dir = os.path.join(utils_dir, 'script')
    if not os.path.isdir(script_dir):
        raise RuntimeError("no script directory found in {}".format(script_dir))
    return script_dir



