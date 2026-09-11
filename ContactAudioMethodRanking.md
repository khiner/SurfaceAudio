# Surface Contact Audio Method Ranking

Methods are ordered by physical plausibility, audible results, performance, and completeness for general surface-contact audio.
It reflects engineering judgment from papers and implementations, with comparable benchmarks unavailable.

| Rank | Method | Scope and limits |
|---|---|---|
| 1 | [Agarwal 2021][agarwal] | Geometry-driven scraping/rolling with listening studies, limited by simplified mechanics and missing inputs |
| 2 | [SDT][sdt] | Broad contact models and reusable resonators with C source, using simplified physics for efficiency |
| 3 | [Conan TASLP 2014][rolling] | Efficient correlated micro-impact rolling with perceptual validation and statistical approximations |
| 4 | [Matusiak 2025][matusiak25] | Passive compliant friction with energy accounting, costly nonlinear solves, and bow-specific assumptions |
| 5 | [Poirot 2023][poirot] | Efficient, perceptually evaluated buzz/rattle synthesis demonstrated for string–obstacle contact |
| 6 | [Conan CMJ 2014][continuous] | Inexpensive rubbing/scratching/rolling transitions with perceptual controls and limited physical calibration |
| 7 | [Matusiak 2024][matusiak24] | Robot-bowing transient validation with fitted parameters and remaining playability discrepancies |
| 8 | Falaize–Roze 2024 | Energy-based coupling with a parameter-dependent [passivity issue][matusiak25] identified by Matusiak 2025 |
| 9 | [Traer 2019][traer] | Economical material-conditioned responses and impacts, with scraping extended by Agarwal 2021 |
| 10 | [Willemsen 2019][willemsen] | Real-time stiff-string friction with body integration and perceptual evaluation left open |
| 11 | [HaTT 2014][hatt] | Measured force/speed-conditioned texture vibration requiring an additional airborne-sound model |
| 12 | [Lagrange 2010][lagrange] | Recorded excitation/resonance analysis with limited predictive geometry-driven rendering |
| 13 | [Lee 2010][lee] | Position-dependent rolling reconstruction requiring individual-contact detection and fitting |
| 14 | [Nakatsuka 2017][nakatsuka] | Deformation-aware friction with substantial microscopic assumptions and limited validation/performance evidence |

Implemented methods have reproduction workflows linked from the [README](README.md).
Willemsen matches published figure traces and executed paper-era source under explicit numerical conventions.
Lagrange uses a disclosed estimator extension; its original synthesis WAVs remain unavailable.
Lee remains queued for implementation.
HaTT remains a haptic-data resource whose results concern contact vibration rather than airborne sound.

## Investigation and implementation queue

Lee remains queued for analysis/synthesis, followed by Nakatsuka friction synthesis.
HaTT supplies a separate data-driven haptic rendering task, with sound radiation outside its published scope.
The Agarwal extensions below remain deferred while their input and parameter availability is unresolved.

### Agarwal 2022 poster and 2025 thesis: continuous-contact forces

Investigate finite micro-impact convolution and fixed-response ramps in thesis Chapters 4–5.
The [poster][poster] uses summed stiffness, while the [thesis][thesis] uses series-equivalent stiffness.
This extension remains queued, with 2021 curvature units and trajectory integration constants unresolved.
[Retained reconstructions](docs/Agarwal.md) identify omitted force terms and compare with matching author examples.

### Agarwal, Traer and McDermott 2023: response fitting

[“Sample-efficient learning of auditory object representations using differentiable impulse response synthesis”][icml] appeared at ICML's DAE workshop.
The ten-mode, ten-noise-band response model, multiresolution fitting, and material sampling are implemented in C++23/Metal.
Twenty author-input fits and material-cohort comparisons have been evaluated.
[Author inputs][icml-data] include 20 training WAVs and 80 generated examples across wood, plastic, metal, and glass.
Downloads and hashes are retained under `references/agarwal/icml2023/`.
Author code, fitted distributions, and the connection to the exact 2021 inputs and 2026 database remain unavailable or unverified.
[The workflow](docs/AgarwalResponseReproduction.md) records comparisons and remaining inputs.

### Agarwal, Traer, Schwartz and McDermott 2026: object acoustics

[“Intuitive knowledge of object acoustics enables perceptual separation of physical variables from impact sounds”][impact] is a bioRxiv preprint.
It was posted January 28, 2026, and remains queued for implementation.
Recover measured responses and fitted material/size distributions, and establish the relationship to Traer 2019.
Implement sampled modal-plus-noise responses and impact-force coupling, then reproduce examples and ablations.
Access to underlying data and parameters remains unverified.
The [response documentation](docs/AgarwalResponseReproduction.md) distinguishes the 2023 and 2026 models.
Local PDFs are available for [2023][icml-local] and [2026][impact-local].

[agarwal]: https://mcdermottlab.mit.edu/scraping_rolling.html
[sdt]: https://github.com/SkAT-VG/SDT
[rolling]: /Users/khiner/physical_audio_papers/continuous_interaction_signal_models/conan2014_rolling.pdf
[matusiak25]: https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1525044/full
[poirot]: /Users/khiner/physical_audio_papers/collision_signal_models/poirot2023_collisions.pdf
[continuous]: /Users/khiner/physical_audio_papers/continuous_interaction_signal_models/conan2014_continuous_interactions.pdf
[matusiak24]: /Users/khiner/physical_audio_papers/elastoplastic_friction/matusiak2024_bowed_string_transients.pdf
[traer]: /Users/khiner/physical_audio_papers/contact_sound_generative/traer2019_rigid_body_contact.pdf
[willemsen]: /Users/khiner/physical_audio_papers/elastoplastic_friction/willemsen2019_stiff_strings.pdf
[hatt]: /Users/khiner/physical_audio_papers/haptic_texture_toolkit/culbertson2014_hatt_paper.pdf
[lagrange]: /Users/khiner/physical_audio_papers/contact_sound_analysis/lagrange2010_sustained_contact.pdf
[lee]: /Users/khiner/physical_audio_papers/contact_sound_analysis/lee2010_rolling_source_filter.pdf
[nakatsuka]: /Users/khiner/physical_audio_papers/friction_sound_physical/nakatsuka2017_adhesion.pdf
[poster]: /Users/khiner/physical_audio_papers/contact_sound_generative/agarwal2022_perceiving_physical_consistency_poster.pdf
[thesis]: https://hdl.handle.net/1721.1/158825
[icml]: https://differentiable.xyz/papers-2023/paper_44.pdf
[icml-data]: https://mcdermottlab.mit.edu/ICML2023/sound_website.html
[impact]: https://doi.org/10.64898/2026.01.28.702236
[icml-local]: /Users/khiner/physical_audio_papers/contact_sound_generative/agarwal2023_differentiable_impulse_response_synthesis.pdf
[impact-local]: /Users/khiner/physical_audio_papers/contact_sound_generative/agarwal2026_intuitive_object_acoustics.pdf
