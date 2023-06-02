# ORIS Inclusive Sampling

Source code for paper:
### ORIS: Online Active Learning Using Reinforcement Learning-based Inclusive Sampling for Robust Streaming Analytics System

---
Requirements:
* Python 3.7
* PyTorch (stable version from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))
* HuggingFace
  * Transformers ([https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index))
  * Datasets ([https://huggingface.co/docs/datasets/index](https://huggingface.co/docs/datasets/index))
* Scikit-learn ([https://scikit-learn.org/stable/install.html](https://scikit-learn.org/stable/install.html))
* SciPy ([https://scipy.org/install/](https://scipy.org/install/))
* NumPy ([https://numpy.org/install/](https://numpy.org/install/))
* tqdm ([https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm))

---
Before running experiments, please load data and model:

Datasets (HuggingFace): [(Twitter and Reddit)](https://gmuedu-my.sharepoint.com/:f:/g/personal/rpandey4_gmu_edu/EmRk6iPMklJGv1JmiRlAOK8Bh_o1B8-qj6uGwoDLJV8ZDA?e=sck4y9)

Model: [ORIS (delta=8)](https://gmuedu-my.sharepoint.com/:f:/g/personal/rpandey4_gmu_edu/Ek6-2ZYHIK5FpFA8vBdion0B52Zra3u7Rp0W_zE7IE88sQ?e=Xkv8cY)

Password: `cikm2023`

`Store both data and model in the home directory`

---
Steps to replicate the experiments:
1. Create a virtual environment `python -m venv venv`
2. Activate virtual environment `source venv/bin/activate`
3. Install and verify all packages mentioned before: `cat requirements.txt | xargs -I {} pip install {}`
4. Make the experiments script executable: `chmod +x run_experiments.sh`
5. Run the script to get the result: `./run_experiments.sh`
