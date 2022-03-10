---
title: 'Asterion: A Python package for fitting asteroseismic acoustic glitches'
tags:
  - Python
  - astronomy
  - solar-like oscillators
  - asteroseismology
authors:
  - name: Alexander J. Lyttle
    orcid: 0000-0001-8355-8082
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: University of Birmingham, Birmingham, United Kingdom, B15 2TT
   index: 1
 - name: Stellar Astrophysics Center, University of Aarhus, Denmark
   index: 2
date: 24 February 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Asteroseismic oscillation modes in solar-like oscillators are sensitive to
changes in their interior structure. Some of these changes manifest as
so-called acoustic glitches (decaying oscillations) in frequency space. For
example, such glitches arise from the second ionisation of helium and the base
of the convective zone. The amplitude of the former correlates with the
near-surface helium abundance, a difficult quantity to measure in low- to
intermediate-mass stars. However, these glitches are difficult to model because
of the small number of observations and poorly defined functional form of the background. We present Asterion, a Python package which uses a Gaussian Process
to model the smoothly varying component of the mode frequencies, $\nu$, as a
function of their radial order, $n$, thus leaving behind the glitches.

# Statement of need

Asteroseismology, the study of the oscillation of stars, has lead to a deeper
understanding of the stellar interior with the advent of high-cadence
space-based photometric surveys. Consequently, high-resolution asteroseismic
data is testing the limits of stellar modelling, with systematic uncertainties
arising from the poor understanding of helium abundance not directly measurable
in low-mass, solar-like stars...

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References