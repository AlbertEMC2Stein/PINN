import os
from os.path import sep

if __name__ == "__main__":
    os.system("pdoc3 --html --config latex_math=True --output-dir . .." + sep + "PDESolver")

    assert os.getcwd().split(sep)[-1] == "docs"

    for file in os.listdir("PDESolver"):
        os.rename(os.getcwd() + sep + "PDESolver" + sep + file, os.getcwd() + sep + file)

    os.rmdir("PDESolver")

