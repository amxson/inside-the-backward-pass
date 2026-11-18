# Inside the Backward Pass

This iCog Labs capstone implements a small neural network engine with NumPy and explains how each operation contributes to backpropagation. It uses a fixed hand-calculation example, numerical gradient checks, and a reproducible XOR experiment.

[Open in Colab](https://colab.research.google.com/github/amxson/inside-the-backward-pass/blob/main/Inside_the_Backward_Pass.ipynb)

## Capstone phases

1. **Derive.** Trace a fixed 2 → 2 → 1 sigmoid network by hand, from the weighted sums and mean squared error to every weight, bias, and input gradient. Compare the reference values with the engine.
2. **Implement.** Build affine layers, sigmoid, ReLU, tanh, mean squared error, and binary cross-entropy. Each layer caches what its backward pass needs. The network applies local derivatives in reverse order and exposes parameter gradients for full-batch gradient descent.
3. **Verify and train.** Perturb every trainable parameter in the checked networks one at a time and compare central differences with backpropagation. Train a linear classifier and an eight-unit tanh network on the same XOR data.

## Executed results

The Phase 1 calculation matched the engine to within 4.75 × 10⁻¹³. All 62 parameter checks passed the 10⁻⁶ threshold. The largest parameter relative error was 4.14 × 10⁻⁹. The largest activation or loss probe error was 3.70 × 10⁻⁸.

Both classifiers took 400 full-batch steps at learning rate 0.15 with 128 training points and 64 separately generated test points.

| Model | Initial training BCE | Final training BCE | Training accuracy | Held-out BCE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear baseline | 0.740204 | 0.693147 | 50% | 0.693147 | 50% (32/64) |
| Tanh network | 0.694577 | 0.058951 | 100% | 0.057776 | 100% (64/64) |

The tanh model learned a curved decision boundary that separated the four XOR groups in this experiment. The linear model’s threshold is a straight line. The notebook contains the data scatter, loss curves, and decision regions.

## Run the notebook

**Local dependencies.** Use Python 3 with NumPy and Matplotlib.

```bash
python -m pip install -r requirements.txt
```

- **VS Code.** Install the Python and Jupyter extensions. Run `python -m pip install ipykernel` in the same environment, open the notebook, select that Python kernel, and choose **Run All**.
- **JupyterLab.** Run `python -m pip install jupyterlab` if needed. Start it with `jupyter lab`, open the notebook, then restart the kernel and run all cells.
- **Google Colab.** Use the link above, connect to a Python CPU runtime, and choose **Runtime → Run all**.

The dataset is generated in the notebook. Rerunning the training cell recreates both models from their seeded initial parameters.

## Repository files

| File | Purpose |
| --- | --- |
| [Inside_the_Backward_Pass.ipynb](Inside_the_Backward_Pass.ipynb) | Derivations, implementation, executed checks, and XOR experiment |
| [Inside_the_Backward_Pass_Report.pdf](Inside_the_Backward_Pass_Report.pdf) | Technical report covering the derivations, gradient verification, and XOR results |
| [requirements.txt](requirements.txt) | Numerical and plotting dependencies |
| [.gitignore](.gitignore) | Excludes local Python, Jupyter, and editor artifacts |
