# <h1 style="font-size: 3.2em; font-weight: bold;">Loss-Oriented Hazard Consistent Incremental Dynamic Analysis (LOHC-IDA)</h1>



This repository provides a single Python script implementing the **LOHC-IDA** framework to simulate hazard-consistent engineering demand parameters (HC-EDPs) using incremental dynamic analysis (IDA). The script is intentionally kept simple so users can easily follow the methodology and map each step to the equations in our paper.

There is a document named **`LOHC_IDA_example.pdf`** with a broad overview of this method and an example calculation demonstrating the application of the LOHC-IDA framework. The results shown here are consistent with the outputs from the Python code.

<h2 style="font-size: 2em; font-weight: bold;"> Reference (Please Cite)</h2>
If you use this code, please cite:  
Sistla S., Chandramohan R., Sullivan T. J. Loss-oriented hazard-consistent incremental dynamic analysis.  
Structural Safety, 2026, Article 102692.  
[https://doi.org/10.1016/j.strusafe.2026.102692](https://doi.org/10.1016/j.strusafe.2026.102692)

## <h2 style="font-size: 2em; font-weight: bold;">What the Code Does</h2>

- Fits multivariate log-linear regression models to IDA results
- Computes hazard-consistent EDP means and variances  
- Preserves EDP correlation structure from IDA
- Simulates correlated HC-EDPs including modelling uncertainty


The equation numbers in the Python script and example document match those in the reference paper to make it easy to follow and understand the method.

<h2 style="font-size: 2em; font-weight: bold;"> Inputs</h2>

The script expects the following Excel files in the **`data/`** directory:

| File | Description |
|------|-------------|
| `EDPs.xlsx` | EDPs from IDA |
| `GM_IMs.xlsx` | Secondary IMs (SaRatio, PGA) |
| `Tar_IM_mean.xlsx` | Mean of target secondary IMs |
| `Tar_IM_cov.xlsx` | Covariance of target secondary IMs |

## <h2 style="font-size: 2em; font-weight: bold;">Usage</h2>
1. Place all input files in the `data/` directory
2. Modify modelling uncertainty at the top of the script if needed
3. Run the code: `python LOHC_IDA.py`

## <h2 style="font-size: 2em; font-weight: bold;">Output</h2>
- **`simulated_demands`** – HC-EDPs in linear space  
These realizations are suitable for loss and risk assessments.

## <h2 style="font-size: 2em; font-weight: bold;">Notes</h2>
- Provided for research and educational use
- See the accompanying calculation document for full theory
- Users are responsible for verifying applicability
- For commercial purposes, please contact me at [saiteja802@gmail.com](mailto:saiteja802@gmail.com) for licensing options.
