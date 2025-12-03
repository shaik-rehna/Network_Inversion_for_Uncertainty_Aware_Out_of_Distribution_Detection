# Network Inversion for Uncertainty Aware Out-of-Distribution Detection 
EE451 Supervised Research Exposition 

* This project implements **Network Inversion**(Paper: ”Network Inversion and its Applications” (https://arxiv.org/pdf/2411.17777))  as a mechanism to generate synthetic Out-of-Distribution (OOD) samples and use them to train a classifier with an explicit **garbage class**. The goal is to improve the classifier’s robustness and uncertainty awareness when encountering inputs outside the training distribution.

* The pipeline iteratively generates OOD samples via a generator trained with inversion losses, appends them to the training set, retrains the classifier, and evaluates performance across various OOD datasets.

---

## **Project Overview**

Modern neural networks often assign high confidence to OOD inputs. To address this, the following approach is used:

### **1. Train a classifier**

* In-distribution data 
* Extra class (**garbage class**): random Gaussian noise initially
* Loss: Weighted Cross-Entropy (dynamic class balancing)

### **2. Train a generator via network inversion**

The generator is optimized so that:

* The classifier assigns *specific labels* to generated samples
* The generated samples follow diverse conditioning distributions
* Generator loss combines:

  * **KL Divergence Loss**
  * **Cross-Entropy Loss**
  * **Cosine Similarity Loss**
    
### **3. Iteratively expand the dataset**

* After every epoch, OOD samples generated via inversion are added to the garbage class.
* The classifier is retrained incrementally.

### **4. Evaluate OOD Detection**

The classifier is evaluated on OOD datasets for AUROC, AUPR and FPR@95TPR

## Experiments:
* Datasets: MNIST, FMNIST, CIFAR10, SVHN, CIFAR100, TinyImageNet-200
* One vs rest approach is used during the experiments. One of the dataset is considered as in-dist and the rest as OOD datasets.
* Training happens on in-dist data and the evaluation on the OOD datasets.
  
Example: MNIST as in-dist, OOD datasests: FMNIST, CIFAR10, SVHN, CIFAR100, TinyImageNet-200
* The code for the experiments are in the folder: [Code](./CODE)


## **Results**

* Below are the results for MNIST as in-dist and FMNIST, CIFAR10, SVHN, CIFAR100, TinyImageNet-200 as OOD datasets

| OOD Dataset          | AUROC  | AUPR   | FPR@95TPR | 
| -------------------- | ------ | ------ | --------- | 
| **FashionMNIST**     | 0.9784 | 0.9719 | 0.0747    | 
| **CIFAR-10**         | 0.9704 | 0.9620 | 0.1137    |
| **SVHN**             | 0.8381 | 0.9152 | 0.4550    |
| **CIFAR-100**        | 0.9626 | 0.9526 | 0.1430    | 
| **TinyImageNet-200** | 0.9692 | 0.9615 | 0.1204    |

* **Avg AUROC:** **0.9437**
* **Avg AUPR:** **0.9526**
* **Avg FPR@95TPR:** **0.1814**


#### Detailed explanation and the results for all the experiments are mentioned in the report: [Report](./EE451_SRE_Report.pdf)







