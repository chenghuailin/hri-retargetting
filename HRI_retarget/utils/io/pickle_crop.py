import pickle 
import os
from HRI_retarget import DATA_ROOT
with open(os.path.join(DATA_ROOT,"motion/galbot/SG/058_clip_semantic_results.pickle"), "rb") as file:
    data = pickle.load(file)

data["angles"] = data["angles"][120:]

with open(os.path.join(DATA_ROOT,"motion/galbot/SG/058_clip_semantic_results_crop.pickle"), "wb") as file:
    pickle.dump(data, file)
