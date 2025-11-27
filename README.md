# Dentistry Patient Action Analysis - SmilePass

## Abstract
This project was initiated from raw data provided by SmilePass and aims to generate innovative insights from limited resources. As a registered research assistant in the MIE program at the University of Toronto, I provided weekly progress updates and reports. The project is supervised by Professor Chi-Guhn Lee and Dhavalkumar Patel. I collaborated with Ruiwu Liu and Prithvi Seran, who also contributed significantly to this research — my sincere thanks to them for their efforts.  

The project now encompasses several research directions aligned with dentists' clinical interests and requirements. My main contributions are in three key areas: **Co-occurrence Analysis**, **Risk Level Modelling**, and **Recommendation System Development**.

## Co-occurrence Analysis
- Analyzed treatment patterns using `procedure_code` within 3- and 6-month rolling time windows.
- Conducted both co-occurrence and sequential occurrence analyses to uncover treatment relationships.
- Applied the Apriori algorithm to compute support and lift values.
- Mapped procedure codes to descriptions and visualized results using lift heatmaps.

## Risk Level Analysis
- Engineered features from treatment history to construct a risk map and patient health scoring system.
- Performed correlation analysis and feature importance evaluation for outcome prediction.
- Automated the pipeline for modelling treatment risk, focusing on procedures like crowns and root canals.

## Recommendation System
- Developed data cleaning and encoding pipelines to prepare for modelling.
- Segmented patient data based on visit consistency and treatment history length.
- Applied unsupervised learning (clustering) to identify patient groups.
- Implemented deep learning models tailored to consistent patient segments to generate treatment recommendations.
