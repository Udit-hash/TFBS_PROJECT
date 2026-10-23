| Method | Sequences | Core AUROC | Peak on motif | Importance on motif | ms / sequence |
|---|---|---|---|---|---|
| Gradient (saliency) | 500 | 0.821 ± 0.219 | 67.8% | 22.3% | 1.0 |
| Gradient × input | 500 | 0.759 ± 0.194 | 69.8% | 27.0% | 0.6 |
| ShiftSmooth | 500 | 0.842 ± 0.219 | 69.4% | 22.3% | 3.3 |
| Integrated Gradients (flat 0.25 reference) | 500 | 0.714 ± 0.189 | 56.2% | 21.3% | 21.2 |
| Integrated Gradients (shuffled references) | 500 | 0.722 ± 0.185 | 58.6% | 17.5% | 83.3 |
| Random (chance level) | 500 | 0.499 ± 0.097 | 14.0% | 9.9% | 0.0 |

Motif MA0478.2 covers 9.9% of each sequence, so 'importance on motif' above that means the method concentrates on the real site.
