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

Model directory contains trained ORIS model.

Data directory contains filtered Twitter and Reddit data adapted from HuggingFace Datasets.

Original data: 
* Twitter ([https://huggingface.co/datasets/dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion))
* Reddit ([https://huggingface.co/datasets/go_emotions](https://huggingface.co/datasets/go_emotions))

---
Steps to replicate the experiments:
1. Create a virtual environment `python -m venv venv`
2. Activate virtual environment `source venv/bin/activate`
3. Install and verify all packages mentioned before: `cat requirements.txt | xargs -I {} pip install {}`
4. Make the experiments script executable: `chmod +x run_experiments.sh`
5. Run the script to get the result: `./run_experiments.sh`
