from Analyze import analyze
import os



if __name__ == '__main__':
    """
    For each dataset in the datasets fold executes the analysis
    """
    root = 'Data'
    for file in os.listdir(root)[:-1]:
        print("ANALYSIS FOR "+file+"")
        print("-" * 50)
        analyze(file=file, folds=5)
        print("-" * 50)