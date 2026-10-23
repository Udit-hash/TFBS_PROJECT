# DNA shape feature: a simple explanation

*For explaining this feature to the mentor or in the viva. No code knowledge needed.*

## The one-minute version

> Our main model reads the **letters** of the DNA (A, C, G, T). But the letters also decide the DNA's **physical shape**: how wide its groove is and how twisted each step is. Some proteins recognise their binding spot partly by **feeling that shape**, not only by reading letters. So we used a published tool, **DNAshapeR**, to compute 4 shape values at every position and gave them to the model as a second input. Then we trained the model **with and without** shape on the same data to see whether shape improves prediction.

## Why shape matters (the idea)

A transcription factor recognises DNA in two ways:

1. **Reading the letters.** Parts of the protein touch the edges of the bases inside the DNA's grooves, a bit like reading Braille.
2. **Feeling the shape.** Some letter combinations make the DNA narrower, wider, stiffer or more bent. A protein can fit into a certain shape like a key into a lock. For example, runs of A and T tend to make the *minor groove* narrower, which some proteins prefer.

This was shown in a well-known paper: Rohs et al., *"The role of DNA shape in protein–DNA recognition"*, Nature, 2009. DeepSTF, which our paper cites, also combined sequence and shape.

## The 4 shape values we use

| Short name | Full name | What it means in plain words |
|---|---|---|
| **MGW** | Minor groove width | How wide the narrow groove of the helix is at that spot, in ångströms |
| **ProT** | Propeller twist | How much the two bases in one pair twist against each other, like the blades of a propeller |
| **Roll** | Roll | How much one base pair tilts open against the next one, which bends the DNA |
| **HelT** | Helix twist | How far the helix rotates from one base pair to the next (normally about 34°) |

MGW and ProT belong to **each base pair** (101 values for 101 bp). Roll and HelT belong to **each step between two pairs** (100 values). That's why the website's shape table has a few empty cells.

## How DNAshapeR gets the values

It doesn't measure anything in a lab, and it doesn't run a slow simulation each time. Scientists ran detailed physics simulations of DNA once and stored the results in a **lookup table**. For each position, DNAshapeR looks at the **5 letters centred on it** (a "pentamer") and looks up the shape value for that 5-letter combination. That makes it fast and repeatable. Because it needs 2 letters on each side, the very first and last positions can't get a value.

## How the model uses it

```
DNA letters (101 bp) ──► CNN + BiLSTM + Transformer ──► 240 numbers ─┐
                                                                     ├──► combine (304) ──► probability of binding
4 shape tracks ───────► small CNN ──────────────────► 64 numbers  ─┘
```

The two parts are trained **together**, so the model learns how much to trust each one.

## How we test whether it helps

In the training notebook (Part B), the shape model is trained on the **same 5 data splits** as the other models and scored on the **same untouched test set**. If its average AUROC is higher than the letter-only model's, shape helped. If not, that's still a valid, honest result: *"for FOSL2, shape added no information beyond the sequence."* (FOSL2 is a bZIP protein that mostly reads letters in the major groove, so a small or zero gain wouldn't be surprising.)

## Likely questions from the mentor

**Q: Are these shape values measured experimentally?**
A: No, they're predictions from DNAshapeR's lookup table, which was built from physics simulations. They describe the DNA's typical shape on its own, not how a protein bends it.

**Q: Why 4 features when the paper says 5?**
A: The paper also listed electrostatic potential. We used the 4 classic DNAshapeR features. We'll update the paper to say 4. Electrostatic potential could be added later.

**Q: Why not just use the letters? The model can learn shape from letters anyway.**
A: In principle, yes. Shape is fully determined by the letters. But giving it explicitly can help the model learn faster or with less data. That's exactly what the comparison tests.

**Q: Does the website need this?**
A: No. The website uses the letter-only model by default, because the shape model needs the R software installed. The shape matrix button is an extra tool for exploring a sequence.

## Where it lives in the code

- `ml/dna_shape.py`, `ml/dna_shape.R`: call DNAshapeR and read its output
- `ml/prepare_shape_dataset.py`: computes the shape values for the whole dataset
- `ml/models/shape_fusion.py`: the combined letters + shape model
- `ml/train_all.py --with-shape`: trains and compares it (Part B of the Colab notebook)
