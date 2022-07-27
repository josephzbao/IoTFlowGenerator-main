import glob
import subprocess

extended = "../save/*/"
paths = glob.glob(extended)
counter = 0
for path in paths:
    positive_file = path + 'real_data.txt'
    negative_file = path + 'generator_sample.txt'
    eval_file = path + 'eval_file.txt'

    with open('seq1.txt', 'r') as f1, open('seq2.txt', 'r') as f2, open(str(counter) + 'seq3.py', 'w') as f3:
        start = f1.read()
        start += "\n" + "positive_file = " + "\"" + positive_file + "\""
        start += "\n" + "negative_file = " + "\"" + negative_file + "\""
        start += "\n" + "eval_file = " + "\"" + eval_file + "\"" + "\n"
        start += f2.read()
        f3.write(start)
        script = ["python2.7", str(counter) + "seq3.py"]
        process = subprocess.Popen(" ".join(script),
                                   shell=True,
                                   env={"PYTHONPATH": "."})

