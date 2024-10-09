---
title: 'ExpFamilyPCA.jl: A Julia Package for Exponential Family Principal Component Analysis'
tags:
  - Julia
  - compression
  - dimensionality reduction
  - PCA
  - exponential family
  - EPCA
  - open-source
  - POMDP
  - MDP
  - sequential decision making
  - RL
authors:
  - name: Logan Mondal Bhamidipaty
    orcid: 0009-0001-3978-9462
    affiliation: 1
  - name: Mykel J. Kochenderfer
    orcid: 0000-0002-7238-9663
    affiliation: 1
  - name: Trevor Hastie
    orcid: 0000-0002-0164-3142
    affiliation: 1
affiliations:
 - name: Stanford University
   index: 1
date: 9 September 2024
bibliography: paper.bib
---

# Summary

Principal component analysis (PCA) [@PCA1; @PCA2; @PCA3] is effective at compressing, denoising, and interpreting normally distributed data, but it struggles with binary, count, and compositional data, which are common in fields like geochemistry, marketing, genomics, and political science, and machine learning [@composition; @elements]. Exponential family PCA (EPCA) [@EPCA] extends PCA to handle data from any exponential family distribution, making it a better fit for these diverse data types.

`ExpFamilyPCA.jl` is a library for exponential family principal component analysis (EPCA) [@EPCA] written in Julia, a dynamic language for scientific computing [@Julia]. This is the first Julia package to implement EPCA and the first in any language to support multiple distributions for EPCA.