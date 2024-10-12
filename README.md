**Drag-Based Image Editing using Diffusion Models**


## Overview
Drag-based image editing is a style of image editing where the user can "drag" any points of the image to precisely reach target points. Successful drag-based editing involves moving the object as well as preserving the other details in the original image. This repository contains the code pertaining to an exploratory project that relies on using feature correspondences in large-scale diffusion models in order to perform Drag-based image editing. 

## Problem Setup
Given an image, the user has the option to specify 2 sets of points. These sets are called the handle points and target points, the goal of drag-based editing is to modify the image in a way such that the part of the image that the handle points refers to moves to the location specified by the target points. Here are a few examples,

<p align="center">
  <img src="teaser_fig/horse.png" alt="Image 1" width="45%" style="vertical-align:top; margin-right:5%;" />
  <img src="teaser_fig/mountain.png" alt="Image 2" width="45%" style="vertical-align:top;" />
</p>

<p align="center">
  <strong>Figure 1:</strong> Description of image 1 on the left. &nbsp;&nbsp;&nbsp;&nbsp; 
  <strong>Figure 2:</strong> Description of image 2 on the right.
</p>

![Screen Shot 2024-06-05 at 4 06 31 PM](https://github.com/karths8/K-Specialist-Approach-to-Code-Generation/assets/47289950/e29a16d4-952f-40b0-bd1d-bd545b2f1b08)

Three Data Generation Strategies we prototyped for our training dataset. **Strategy 1** resulted in many cases of model refusal to use the given
keyword list to produce a meaningful question-code pair. **Strategy 2** generated data points similar to the in-context examples
with slight modifications. **Strategy 3**, a combination of **Strategy 1** and **Strategy 2**, generated complex and
unique data points.

-->

