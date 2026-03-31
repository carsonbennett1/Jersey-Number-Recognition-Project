## Group 14 - Jersey Number Recognition Project - COSC 419/519

## Setup
refer to [setup](docs/SETUP_GUIDE.md)

## Approaches

**NOTE**: ALL APPROACHES ARE DONE IN DIFFERENT BRANCHES WITH DIFFERENT IMPLEMENTATIONS. SEE "BRANCHES AND THEIR CONTRIBUTIONS" SECTION BELOW FOR FURTHER DETAILS.

Approach 1: Multi-Task Learning for Frame-Level Recognition

Approach 2: Reliability-Weighted Top-L Consolidation

Approach additional: Fine tuned parameters and fixed motion blur


#### Member Roles
| Name | Roles, Contribution |
|------|---------------------|
| Carson Bennett  | Proposed approach 1, worked on implementing approach 2, involved in improving upon approach 2, recorded content for video demo |
| Aakash Tirathdas  | Helped with initial setup, worked on implementing on approach 1, did an additional approach in experimental, recorded content for video demo |
| Harjot Sahota  | helped with initial setup, worked on implementing approach 2, recorded progress for different implementations, recording content for video demo |
| Brendan James  | Researched for proposal, worked on evaluating results, editing for video demo |
| Zhongda  | Coordinated and drafted the proposal, presentation, and report; designed the original Top-L consolidation method |


#### Branches and their Contributions

| Approach | Variation | Author(s) | Accuracy | Branch Link |
|----------|-----------|-----------|----------|-------------|
| Main/raw repo | Base Repo | Aakash, Harjot | 86.70520231213872% | base github link |
| Approach 1 | Multi-Task Learning for Frame-Level Recognition | Aakash |  85.2188274153592% | [here](https://github.com/carsonbennett1/Jersey-Number-Recognition-Project/tree/approach1) |
| Approach 2 | Vanilla Implementation | Carson, Harjot | 83.3195706028076% | [here](https://github.com/carsonbennett1/Jersey-Number-Recognition-Project/tree/top-L) |
| Approach 2 | Top-L: L as Tracklet Frame Length Percentage | Carson | 85.2188274153592% | [here](https://github.com/carsonbennett1/Jersey-Number-Recognition-Project/tree/top-L-improvements) |
| Approach 2 | Top-L whole number instead of 10's digit and ones digit | Harjot | 84.31048720066062% | [here](https://github.com/carsonbennett1/Jersey-Number-Recognition-Project/tree/harjot/Top-L) 
| Additional Approach | Base repo tuning and fixing | Aakash | 88.52188274153592% | [here](https://github.com/carsonbennett1/Jersey-Number-Recognition-Project/tree/experimental) |
| Batching Base | Add batching to the base branch | Aakash, Brendon | - | [here](https://github.com/carsonbennett1/Jersey-Number-Recognition-Project/tree/batches) |

## Video Demo
https://youtu.be/-VSREV-4DrI

