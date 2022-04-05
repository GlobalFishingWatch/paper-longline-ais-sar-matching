# paper-longline-ais-sar-matching
A Repo for the paper `Revealing the Global Longline Fleet with SAR`

<!-- #region -->
This repo contains the methods and analysis used to develop the Revealing the Global Longline Fleet with SAR paper. It has the following folders with the following contents:

### 01_Acquisitions 
The data obtained from KSAT on SAR detections and scenes, and the code to process the files they provided into csv and BigQuery tables.

### 02_Evaluation_of_Fishing_Activity 
 - map of longline activity with areas of interst
 - analysis of fishing activity in these regions from AIS
 - bar charts showing activity by flag state in these regions
 - size distribution of vessels in each region
 
### 03_Vessel_Review
We reviewed vessels to be sure of the vessel classification, and to make sure we were correclty classifying which MMSI were gear and which were not. This includes a good deal of manual reviews, and plotting of vessel tracks.

### 04_Probability_Raster_Generation
 - Code that produces probability rasters 

### 05_SAR_AIS_Matching
This includes steps that match SAR detections to AIS:
 - extrapolationg vessels to the time of the image
 - estimating the likelihood a given vessel is within a scene, based on the probability rasters
 - scoring each detection to vessel pair using a) a weighted average of the probability rasters before and after and b) an average of the before and after raster
 - a ranking method to decide on the most likely matches

### 06_Evaluating_Matching
 - Comparison of multiplying versus averaging scores before and after the scenes
 - Visual comparisons of ambiguous matches and low-confidence matches
 - Figures for the paper that demonstrate the rasters
 - An estimation of how long between the image and the detection. 

### 07_Dark_Vessel_Estimates
Estimate the relationship between vessel size from GFW and SAR, and recall as a function of length. These are combined to produce a probabilistic estimate of the number of non-broadcasting fishing vessels in the region.

### 08_Area_of_Ocean_to_be_Monitored
Using this technique, how much of the ocean could we monitor? This shows the amount of imagery needed to track 5%, 10%, 20% and 50% of the global pelagic longline fleet. We also estimate the same for for the area that is not covered by Sentinel-1 imagery.
