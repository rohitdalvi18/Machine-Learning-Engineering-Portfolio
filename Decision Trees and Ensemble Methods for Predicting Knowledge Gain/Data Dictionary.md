# SAL-Lightning Dataset Dictionary

This document describes the variables in the SAL-Lightning combined dataset (**`sal_lightning_combined.csv`**), which contains [data from a study](https://data.uni-hannover.de/dataset/sal-dataset) where participants learned about the formation of lightning and thunder through web searches.

## Participant Information

| Variable | Description |
| :---- | :---- |
| **`p_id`** | Participant ID \- Unique identifier for each participant |
| **`d_sex`** | Gender of participant (1 \= female, 2 \= male) |
| **`d_age`** | Age of participant |
| **`d_field_of_study`** | Field of study of the participant |
| **`d_no_sem`** | Number of semesters completed by participant |
| **`d_lang`** | First language of the participant |

## Knowledge Assessment

| Variable | Description |
| :---- | :---- |
| **`k_mc_sum_t1`** | Number of correct multiple-choice questions before web search (pretest) |
| **`k_mc_sum_t2`** | Number of correct multiple-choice questions after web search (posttest) |
| **`kg_mc`** | Knowledge gain in multiple-choice test (t2 \- t1) |
| **`essay_C1`** | Number of correct concepts in essay before web search |
| **`essay_C2`** | Number of correct concepts in essay after web search |
| **`KG_essay`** | Knowledge gain in essay (essay\_C2 \- essay\_C1) |

## Cognitive Abilities

| Variable | Description |
| :---- | :---- |
| **`LGVT_speed`** | Reading speed \- Number of words read in standardized reading test |
| **`LGVT_score`** | Reading comprehension score \- Points for correctly solved sentences |
| **`WMC_Recalls`** | Working memory capacity \- Number of correctly recalled sets |
| **`WMC_Sentence`** | Working memory capacity \- Number of correctly solved sentences |
| **`CRT_sum`** | Cognitive reflection \- Number of correctly solved cognitive reflection tasks |
| **`DSSQ_mean`** | Task engagement \- Mean score on the Dundee Stress State Questionnaire |

## Web Search Behavior

| Variable | Description |
| :---- | :---- |
| **`session_duration`** | Total duration of the web search session in seconds |
| **`num_pages_visited`** | Number of web pages visited during the session |
| **`unique_domains_visited`** | Number of unique domains visited during the session |
| **`num_clicks`** | Number of mouse clicks recorded during the session |
| **`num_scrolls`** | Number of scroll events recorded during the session |
| **`total_interactions`** | Total number of interaction events recorded |
| **`num_browser_tabs`** | Number of browser tabs opened during the session |
| **`avg_time_active_per_tab`** | Average time (in seconds) a tab was active |

## 

## Derived Metrics

| Variable | Description |
| :---- | :---- |
| **`avg_time_per_page`** | Average time spent per page in seconds (session\_duration / num\_pages\_visited) |

## Notes

* This dataset combines information from multiple sources: knowledge tests, and browsing behavior

* The knowledge gain variables (**`kg_mc`** and **`KG_essay`**) represent the difference between post-test and pre-test scores

