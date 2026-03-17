# Mushroom Toxicity Detection — Naive Bayes from Scratch

I built this project to understand how Naive Bayes actually works under the hood, not just calling sklearn and getting an accuracy score. So I wrote the entire thing from scratch using only NumPy and Pandas.

The goal is simple — given a mushroom's physical features, figure out if it's safe to eat or poisonous.

---

## Why this project?

I wanted to understand what's really happening inside a Naive Bayes classifier. Things like:
- What does "prior probability" actually mean with real data?
- Why do we multiply all those probabilities together?
- What happens when a probability is zero?

This dataset was a great fit because all features are categorical and the problem is straightforward — edible or poisonous.

---

## Dataset

UCI Mushroom Dataset — 8124 mushrooms, 22 features like odor, cap shape, gill size, bruises etc.

The target column is `class`:
- `e` → edible
- `p` → poisonous

All features are letter-coded (e.g. odor: `p` = pungent, `f` = foul, `a` = almond) so I used LabelEncoder to convert them to numbers before training.

---

## What I implemented

**prior_prob()** — calculates how common each class is in the training data. For this dataset it came out to roughly 51% edible and 48% poisonous, almost balanced.

**cond_prob()** — the likelihood part. Given that a mushroom belongs to a class, what fraction of them have a specific feature value? There's a small fix here where instead of returning 0 when a feature is never seen, I return a tiny value (1e-6). Without this, a single zero wipes out the entire score for that class.

**predict()** — combines everything. Calculates a posterior score for each class by multiplying the prior with all 22 likelihoods, then picks the class with the highest score.

**accuracy()** — runs predict() on every row in the test set and compares against true labels.

---

## Results

Got 98.5% accuracy on the test set which I was honestly surprised by, given how simple the math is.

The odor feature turned out to be the strongest signal by far. If a mushroom smells pungent, foul, or spicy — it's almost always poisonous. Musty or almond smell tends to mean edible. Some odor values never appear in edible mushrooms at all, which is why the zero handling in cond_prob matters so much.

---

## A bug I ran into

Early on I was getting exactly 0.5 accuracy. Turned out to be two problems:

First, the return statement inside predict() was indented inside the for loop, so it returned after calculating only the edible score and never got to poisonous.

Second, no zero handling in cond_prob(). When any feature probability comes out 0.0, multiplying it in makes the whole likelihood score 0.0. Both classes end up at zero, argmax always picks index 0 (edible), and you get roughly 50% accuracy on a balanced dataset.

Fixing the indentation and adding the 1e-6 fallback brought it straight up to 98.5%.

---

## How to run

```bash
pip install numpy pandas scikit-learn
python mushroom_toxicity_detection.py
```

---

## Files

- `mushrooms.csv` — the dataset
- `mushroom_toxicity_detection.py` — all the code
