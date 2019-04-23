import numpy as np
from caltech42 import Caltech42

model_1 = np.load("0.9640.npy")
model_2 = np.load("0.9671.npy")
model_3 = np.load("0.9641.npy")
model_4 = np.load("adversarial_test.npy")
caltech42 = Caltech42()

# Generate test set annotations, but in args.logdir to allow parallel execution.
with open("caltech42_competition_test.txt", "w", encoding="utf-8") as out_file:
    for i in range(caltech42.test.size):
        p = model_1[i] + model_2[i] + model_3[i] + model_4[i]
        print(np.argmax(p), file=out_file)
        #print("{}: {}".format(i, caltech42.labels[np.argmax(p)]), file=out_file)