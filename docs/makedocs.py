import os
from os.path import sep

if __name__ == "__main__":
    if os.getcwd().split(sep)[-1] != "docs":
        raise PermissionError("Run this script from the docs folder.")
    
    os.system("py -3.9 -m pdoc --html --config latex_math=True --output-dir . .." + sep + "PDESolver")

    for file in os.listdir("."):
        if file.endswith(".html"):
            os.remove(file)

    for file in os.listdir("PDESolver"):
        if file.endswith(".html"):
            os.rename(os.getcwd() + sep + "PDESolver" + sep + file, os.getcwd() + sep + file)

    os.rmdir("PDESolver")

