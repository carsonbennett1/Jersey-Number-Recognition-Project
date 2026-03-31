## Top-L: L as Tracklet Frame Length Percentage - Improvements to Approach 2

Determined to try and get the most out of our proposal, I came up with an idea that extended our base implementation. When doing trial and error with our L values, I realized that we were using a constant L value with an unknown number of frames for each tracklet. To further explain, our constant L was gathering different top-L frames based on how many frames the tracklet had.  

For example, suppose we have two traklets, and a constant L of 7.
- Tracklet A: 10 Frames, so L of 7 will return the 7 best predictions for that digit in that respective frame (70%)
- Tracklet B: 750 Frames, so L of 7 will return the 7 best predictions for that digit in that respective frame (0.9%)

As you can see based on the percentage, using a constant L can end up A), taking too many confidence values, so poor confidence values are included as well, or B), not taking enough predictions to result in an adequate understanding of the confidence for that digit.
Instead, I decided to take the size of all frames in a tracklet, and grab a percentage of the size to be used as our top L. (For example, get tracklet frame length, and grab 30%, 50%, or 70% of our top confidence value frames)

For example, suppose we have two tracklets, and a constant L of 30%
- Tracklet A: 10 Frames, so L of 0.3 will return the 3 best predictions for that digit in that respective frame (30%)
- Tracklet B: 750 Frames, so L of 0.3 will return the 225 best predictions for that digit in that respective frame (30%)

Results:

| Method | L | qt | Accuracy | Notes |
|--------|---|----|----------|-------|
| Baseline | N/A | N/A | 86.95293146160198% | N/A |
| Top-L simple | 3 | 1 (before using legibility scores) | 83.3195706028076% | Simple version that plugs in our formula into the baseline method |
| Top-L: L as Tracklet Frame Length Percentage | 0.7 | Raw_scores | 85.2188274153592% | Instead of having a constant L value, a relative L percentage based on tracklet length has been implemented | 

This slight tweak to the handling of L and our top confidence values resulted in an overall better accuracy score than the vanilla implementation coming in at 85.2188274153592%. While still slightly under the baseline method, we were still able to get a bit more out of the Top-L consolidation approach than the vanilla implementation.
