## Overview of Top-L and Understanding How it can be Implemented

#### Definitions
**Frame**: single picture of the player

**Tracklet**: multiple different frames of a player all taken within milliseconds or seconds of one another. I.e. an array of frames.

#### Legend
| Symbol | Designation | Implementation |
|--------|-------------|----------------|
| T |frames available in tracklet| image_predictions |
| K |candidate numbers (0-99)| columns in tens & units likelihood |
| qt | legibility score for each frame in a tracklet | 1 at the moment (needs to be updated to actual legibility score from files stored after pipeline execution |
| Pt(k) | probability for candidate k in frame t | values inside tens & units likelihood |
| L | how many top frames to keep | L = (whatever we decide to be our number of top frames) |
| ε | small decimal to avoid possibility of log(0)| ε =  1e-9 | 

**Advice**: it's helpful to keep in mind that "candidate" is just jersey number.
- candidate = jersey number

#### General Idea in Plain English
Take all possible jersey numbers, starting at 1, and end when we have passed 99 (no 3 digits in soccer/ European football).

We'll use '1' in the simple passthrough below (Translates to one iteration).

For each frame in the tracklet, calculate v(t,k) which, in english, would be the legibility score of the frame in the tracklet, multiplied by the log of the likelihood that the jersey number in the frame is a 1, plus ε.

v(t,k) = qt x log(Pt(k) + ε)

Next, we'll take the output of v(t,k) and take the top L frames - the frames with the highest confidence values that the jersey number is a 1. Following, use sum to change the array from 2D to 1D and sum the total score by candidate.

Finally, use argmax to get the highest confidence level from the candidate total scores.

So, let's assume that jersey number 1 reported numerous very high confidence ratings. Those confidence ratings over numerous frames would be picked up by Top-L, and summed to get a high score. This makes it likely that an 'argmax' will find this candidate appealing as it is a high decimal number (don't forget we're logging here).

However, let's assume that jersey number 22 reported very few high confidence rating. Those high confidence ratings would still be picked up in the Top-L process, but the final sum of them all might return a negative number at the end. This would not make it an ideal selection by argmax.

#### In Summary
For each jersey number possible, compute frame-by-frame confidence of the tracklet with the most confident values (top-L). If frame-by-frame confidence is low or staggers, this is not our jersey number. However, if the frame-by-frame confidence is relatively high, then this number could likely be the actual one in the tracklet.

#### Main Pipeline Integration
Originally, at step 8 (combine tracklet results), the pipeline calls the method "helpers.process_jersey_id_predictions(...)" which can be found in the helpers.py file (line 611). The method then goes on to call a sub-method "find_best_prediction(...)." Interestingly enough, there are also two other methods that can be swapped in at step 8 that appear to have not made the final cut as seen with "process_jersey_id_predictions_raw(...), and process_jersey_id_predictions_bayersian(...)."

Our new methods will swap out process_jersey_id_predictions(...) and find_best_predictions(...) with the workflow below:

```
main.py
  └──> helpers.process_jersey_id_predictsion_top_L(...)
            └──> predict_jersey_number_top_L(...)
                     └──> candidate_and_frame_processing(...)
```