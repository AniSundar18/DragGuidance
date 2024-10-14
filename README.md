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
  <strong>Figure 1:</strong> Moving the horse tail up. &nbsp;&nbsp;&nbsp;&nbsp; 
  <strong>Figure 2:</strong> Increasing the height of a mountain.
</p>

## Identifying objects using feature correspondence
Drag-based editing relies on 3 important steps,
1. Identifying the object corresponding to the handle point
2. Moving the identified object as per the edit instructions
3. Preserving the identity of the image in other places



