# Human Plasma Protein Binding (PPB) of Compounds

IDL-PPB aims to obtain the plasma protein binding (PPB) values of a compound. Based on an interpretable deep learning model and using the algorithm fingerprinting (AFP) this model predicts the binding affinity of the plasma protein with the compound.

This model was incorporated on 2023-02-03.

## Information
### Identifiers
- **Ersilia Identifier:** `eos22io`
- **Slug:** `idl-ppbopt`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Property calculation or prediction`
- **Biomedical Area:** `ADMET`
- **Target Organism:** `Homo sapiens`
- **Tags:** `Fraction bound`, `ADME`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `1`
- **Output Consistency:** `Fixed`
- **Interpretation:** Fraction of PPB (protein plasma bounding) from 0 to 1. High affinity are fraction of ppb > 0.8, low levels of affinity are fraction of ppb < 0.4

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| ppb_fraction | float | low | fraction of molecule bound to plasma protein (0 to 1) |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos22io](https://hub.docker.com/r/ersiliaos/eos22io)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos22io.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos22io.zip)

### Resource Consumption
- **Model Size (Mb):** `8`
- **Environment Size (Mb):** `4195`
- **Image Size (Mb):** `4211.68`

**Computational Performance (seconds):**
- 10 inputs: `34.72`
- 100 inputs: `27.82`
- 10000 inputs: `698.51`

### References
- **Source Code**: [https://github.com/Louchaofeng/IDL-PPBopt](https://github.com/Louchaofeng/IDL-PPBopt)
- **Publication**: [https://pubs.acs.org/doi/10.1021/acs.jcim.2c00297](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00297)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2022`
- **Ersilia Contributor:** [carcablop](https://github.com/carcablop)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [GPL-3.0-only](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos22io
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos22io
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
