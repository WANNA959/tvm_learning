from tvm.contrib.download import download_testdata
from scipy.special import softmax
import numpy as np
import os.path


# Download a list of labels
labels_path = "synset.txt"

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

output_file = "predictions.npz"

# Open the output and read the output tensor
if os.path.exists(output_file):
    with np.load(output_file) as data:
        # 正数+归一化，概率和=1
        scores = softmax(data["output_0"])
        # 把shape中为1的维度去掉
        scores = np.squeeze(scores)
        # ranks是排序的index
        ranks = np.argsort(scores)[::-1]

        for rank in ranks[0:5]:
            print("class='%s' with probability=%f" %
                  (labels[rank], scores[rank]))
