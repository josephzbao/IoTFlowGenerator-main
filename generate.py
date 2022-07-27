import subprocess
import os

def generate(real_tokens):
    writeTokens(real_tokens, "SeqGAN/save/real_data.txt")
    os.chdir("SeqGAN")
    process = subprocess.Popen("python2.7 sequence_gan.py", shell=True, stdout=subprocess.PIPE)
    while process.poll() is None:
        l = process.stdout.readline()  # This blocks until it receives a newline.
        print(l)
    process.stdout.readline(process.stdout.readline())
    print(process.returncode)
    return readTokens("save/generator_sample.txt")

def writeTokens(real_tokens, location):
    toWrite = "\n".join([" ".join(tokensToString(tokens)) for tokens in real_tokens])
    with open(location, "w") as t:
        t.write(toWrite)
        t.close()

def readTokens(location):
    with open(location, 'r') as f:
        return [stringToTokens(x) for x in f.readlines()]

def tokensToString(tokens):
    return [str(x) for x in tokens]

def stringToTokens(tokensString):
    return [int(x) for x in tokensString.split()]
