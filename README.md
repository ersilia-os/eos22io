# Human Plasma Protein Binding (PPB) of Compounds

IDL-PPB aims to obtain the plasma protein binding (PPB) values of a compound. Based on an interpretable deep learning model and using the algorithm fingerprinting (AFP) this model predicts the binding affinity of the plasma protein with the compound.

## Identifiers

* EOS model ID: `eos22io`
* Slug: `idl-ppbopt`

## Characteristics

* Input: `Compound`
* Input Shape: `Single`
* Task: `Regression`
* Output: `Experimental value`
* Output Type: `Float`
* Output Shape: `Single`
* Interpretation: This model receives smiles as input and returns as output the fraction PPB, which measures the affinity of the binding of the plasma protein. In the analysis of results by the author, they indicate high affinity (fraction of ppb >80%), medium affinity (40% <= fraction of ppb <=80%) and as low levels of affinity (fraction of ppb < 40%). Note: Inorganics and salts are out of the applicability domain of the model, So for these compounds the output is Null.

## References

* [Publication](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00297)
* [Source Code](https://github.com/Louchaofeng/IDL-PPBopt)
* Ersilia contributor: [carcablop](https://github.com/carcablop)

## Citation

If you use this model, please cite the [original authors](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00297) of the model and the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license. The model contained within this package is licensed under a GPL-3.0 license.

Notice: Ersilia grants access to these models 'as is' provided by the original authors, please refer to the original code repository and/or publication if you use the model in your research.

## About Us

The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission!