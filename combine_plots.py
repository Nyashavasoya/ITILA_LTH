import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm


DPI = 1200
prune_iterations = 15
arch_types = ["fc1"]
datasets = ["mnist"]


for arch_type in tqdm(arch_types):
    for dataset in tqdm(datasets):
        d = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/lt_compression.dat", allow_pickle=True)
        b = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/lt_bestaccuracy.dat", allow_pickle=True)
        c = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/reinit_bestaccuracy.dat", allow_pickle=True)

        #plt.clf()
        #sns.set_style('darkgrid')
        #plt.style.use('seaborn-darkgrid')
        a = np.arange(prune_iterations)
        plt.plot(a, b, c="blue", label="Winning tickets") 
        plt.plot(a, c, c="red", label="Random reinit") 
        plt.title(f"Test Accuracy vs Weights % ({arch_type} | {dataset})") 
        plt.xlabel("Weights %") 
        plt.ylabel("Test accuracy") 
        plt.xticks(a, d, rotation ="vertical") 
        plt.ylim(0,100)
        plt.legend() 
        plt.grid(color="gray")

        save_dir = f"{os.getcwd()}/plots/lt/combined_plots"
        os.makedirs(save_dir, exist_ok=True)  # create all intermediate directories if they don't exist

        plt.savefig(f"{save_dir}/combined_{arch_type}_{dataset}.png", dpi=DPI, bbox_inches='tight')
 

        plt.savefig(f"{os.getcwd()}/plots/lt/combined_plots/combined_{arch_type}_{dataset}.png", dpi=DPI, bbox_inches='tight') 
        plt.close()
        #print(f"\n combined_{arch_type}_{dataset} plotted!\n")
