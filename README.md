# Top-L Hard Assignment

## What this is

Our original Top-L method did not beat the baseline accuracy of **86.95%**.

Because of that, I tried a simpler different version of Top-L using **hard assignment**.

My final accuracy with this version was **85.2188274153592%**.

So this version still did not beat baseline, but it was fairly close.

---

## The main idea

The older method treated the tens digit and ones digit separately.

I changed that.

Instead of splitting the number into two separate digit predictions, my method lets each frame vote for **one full jersey number only**.

So if a frame predicts **47**, that frame supports **47 only**.

This is important because the model already predicts the full number, so splitting it into separate digits can lose information.

---

## How my method works

For each frame:
- take the predicted jersey number
- take the confidence
- build a vector for numbers 1 to 99
- give the predicted number the confidence
- give all other numbers a very small value

Then for each tracklet:
- collect all frame predictions
- use the legibility score for each frame
- score each possible jersey number
- keep only the best **L** frame scores
- sum them
- choose the number with the highest score

So the method focuses on the **best frames**, instead of using every frame equally.

---

## Legibility

I also added illegibility handling.

If the frames are too unclear, the tracklet is marked as **-1** instead of forcing a bad prediction.

This happens when:
- the legibility scores are too low
- or the best candidate score is still too weak

This was an important part of my contribution because it helps the system avoid making weak predictions.

---

## Result

Baseline accuracy: **86.95%**  
My hard-assignment Top-L accuracy: **85.2188274153592%**

So overall:
- it did **not** beat the baseline
- but it fixed the digit independence issue
- and it added illegibility detection directly into the final output

---

## Short summary

My version of Top-L was a simpler full-number approach.

Instead of splitting digits, each frame voted for one full number.

Then Top-L kept only the strongest frame evidence.

This gave a result close to baseline, but still a little lower.
