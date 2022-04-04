# prj-dark-targets-walmart
A Repo for the Walmart Dark Targets Report


This repo contains the following analyses in support of the report on dark targets. It has the following:

### Evaluation of Fishing Activity in Areas of Interest
 - map of longline activity with areas of interst
 - analysis of fishing activity in these regions from AIS
 - bar charts showing activity by flag state in these regions
 - size distribution of vessels in each region

### Probability Raster Generation
This includes code that produces probability rasters (will be copied in from elsewhere)

### SAR-AIS Matching 
This includes steps that match SAR detections to AIS. It includes:
 - extrapolationg vessels to the time of the image
 - estimating the likelihood a given vessel is within a scene, based on the probability rasters
 - scoring each detection to vessel pair using a) a weighted average of the probability rasters before and after and b) an average of the before and after raster
 - a ranking method to decide on the most likely matches

### Evaluating Matching
This includes
 - Comparison of multiplying versus averaging scores before and after the scenes
 - Visual comparisons of ambiguous matches and low-confidence matches
 - Figures for the paper that demonstrate the rasters
 - An estimation of how long between the image and the 

### Review of Vessels
We reviewed vessels to be sure of the vessel classification, and to make sure we were correclty classifying which MMSI were gear and which were not. This includes a good deal of manual reviews, plotting of vessel tracks.

### Estimation of dark vessels
Estimate the relationship between vessel size from GFW and SAR, and then also the recall as a function of length. These are combined to produce a probabilistic estimate of the number of non-broadcasting fishing vessels in the region.

### Area of the Ocean to be Monitored
Using this technique, how much of the ocean could we monitor? This shows the amount of imagery needed to track 5%, 10%, 20% and 50% of the global pelagic longline fleet. We also estimate the same for for the area that is not covered by Sentinel-1 imagery.

### Turning your repo into a module
1. If you have not updated the `[metadata]` section of `setup.cfg`, do that now (see step 3 in previous section).
2. Add to dependencies in the `[options]` sections as needed, keeping what is already there so that styling/linting and notebooks are supported unless you are absolutely certain you don't want these.
3. Run `pip install -e .` to install the module. This will create a folder titled `<module>.egg-info` that will allow you to access the code within your `<module>` folder from outside of that folder by doing `import <module>` without any need to use paths.
