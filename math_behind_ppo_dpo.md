# Direct Preference Optimization (DPO) — Complete Mathematical Derivation

This document contains a full mathematical derivation and transcription of the DPO-style loss function based on preference data, as extracted and logically organized from handwritten notes. Each expression is broken down to the smallest logical step to maximize clarity.

---

## 🔹 Step 1: Optimal Policy Definition

We define the optimal policy `p(y|x)` using a reward function `r(y,x)` and a reference policy `\pi_{ref}(y|x)`:

$$
p(y|x) = \exp\left( r(y,x) - \beta \log \pi_{ref}(y|x) \right)
$$

---

## 🔹 Step 2: Solve for the Reward `r(y,x)`

Take the log of both sides:

$$
\log p(y|x) = r(y,x) - \beta \log \pi_{ref}(y|x)
$$

Rearranging terms:

$$
r(y,x) = \log p(y|x) + \beta \log \pi_{ref}(y|x)
$$

Alternatively, using scaling and partition term `\lambda`:

$$
r(y,x) = \beta \log p(y|x) + \lambda
$$

---

## 🔹 Step 3: Difference in Rewards Between Preferred and Rejected Samples

Let `y_w` be the preferred (winner) and `y_l` be the less preferred (loser):

$$
r(y_w, x) - r(y_l, x) = \beta \log \left( \frac{p(y_w|x)}{\pi_{ref}(y_w|x)} \right) - \beta \log \left( \frac{p(y_l|x)}{\pi_{ref}(y_l|x)} \right)
$$

Using log subtraction:

$$
= \beta \log \left( \frac{p(y_w|x) / \pi_{ref}(y_w|x)}{p(y_l|x) / \pi_{ref}(y_l|x)} \right)
$$

Simplify:

$$
= \beta \log \left( \frac{p(y_w|x) \cdot \pi_{ref}(y_l|x)}{p(y_l|x) \cdot \pi_{ref}(y_w|x)} \right)
$$

---

## 🔹 Step 4: Define the Loss Function (Using Sigmoid)

Let:

$$
L = -\log \sigma(r(y_w,x) - r(y_l,x))
$$

Substitute reward difference:

$$
L = -\log \sigma\left( \beta \log \left( \frac{p(y_w|x)}{\pi_{ref}(y_w|x)} \right) - \beta \log \left( \frac{p(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right)
$$

Combine:

$$
= -\log \sigma\left( \beta \log \left( \frac{p(y_w|x) \cdot \pi_{ref}(y_l|x)}{p(y_l|x) \cdot \pi_{ref}(y_w|x)} \right) \right)
$$

---

## 🔹 Step 5: Final Boxed Loss Function (from Notes)

Using y\_1 as winner and y\_2 as loser:

$$
L = -\log \left[ \sigma\left( \beta \cdot \left( \log \frac{n(y_1|x)}{n_{ref}(y_1|x)} - \log \frac{n(y_2|x)}{n_{ref}(y_2|x)} \right) \right) \right]
$$

Or more compact:

$$
L = -\log \left[ \sigma\left( \beta \cdot \log \left( \frac{n(y_1|x) \cdot n_{ref}(y_2|x)}{n(y_2|x) \cdot n_{ref}(y_1|x)} \right) \right) \right]
$$

---

## 🔹 Binary Cross Entropy Derivation

Start from:

$$
-\log \left( \frac{\exp(y_1)}{\exp(y_1) + \exp(y_2)} \right)
$$

Factor denominator:

$$
= -\log \left( \frac{1}{1 + \exp(y_2 - y_1)} \right)
= \log(1 + \exp(y_2 - y_1))
= -\log \sigma(y_1 - y_2)
$$

---

## 🔹 Bradley-Terry Model Probability

$$
P(y_w > y_l | x) = \frac{P(y_w|x)}{P(y_w|x) + P(y_l|x)}
$$

---

## 🔹 Gradient Derivation Sketch

Original objective:

$$
\frac{1}{N} \sum P(y|x) \cdot r(y|x) - \beta \cdot P(y|x) \cdot \log\left( \frac{P(y|x)}{r(y|x)} \right)
$$

Let:

* P(y|x) = f(y|x)
* r(y|x) = r

Then:

$$
= \frac{1}{N} \sum \left[ f(y|x) \cdot r - \beta f(y|x) \log \left( \frac{f(y|x)}{r} \right) \right]
$$

---

## 🔹 Derivative w\.r.t. f(y|x)

Loss:

$$
L = r \cdot f - \beta f \cdot \log \left( \frac{f}{n} \right)
$$

$$
\frac{\partial L}{\partial f} = r - \beta \left( \log \left( \frac{f}{n} \right) + 1 \right)
$$

Set derivative to zero:

$$
0 = r - \beta \left( \log \left( \frac{f}{n} \right) + 1 \right)
$$

$$
\Rightarrow \log \left( \frac{f}{n} \right) = \frac{r - \beta}{\beta}
\Rightarrow f = n \cdot \exp\left( \frac{r - \beta}{\beta} \right)
$$

---

## 🔹 Final Parameter Expression

From the notes:

$$
\theta = \frac{r(y,x) - \beta \cdot \log \left( \frac{f(x)}{n(x)} \right)}{-\lambda}
$$

---

This document completes the transcription of your derivations, expanded fully with each step included clearly.
