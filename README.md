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

## Method
Drag-based editing relies on 3 important steps,
1. Identifying the part of the image corresponding to the handle point
2. Moving the identified object as per the edit instructions
3. Preserving the identity of the image in other places

On a high-level, we first identify the part of the image that needs to be edited. After this, we define an objective which measures the progress of the edit, we optimize this objective while also preserving the other details of the image which gives us an accurate edit. The specifics are given below,

## Identifying the Part to be Dragged in the Image 
We have a single point in our image as the input, we need to use that single point in order to identify the region to be dragged. We do so using feature correspondences. [DIFT](https://arxiv.org/abs/2306.03881) shows that the UNet used in large-scale text-to-image diffusion can be used to establish correspondences between real images. Our key idea is simple, use the feature vector in the UNet corresponding to the handle point, then identify the regions that are most similar (using cosine similarity). The pipeline can be found below,

<div align="center">

![Object Identification](teaser_fig/teaser_dift.png)

*The handle point of the beaks in one of the birds can be used to detect the beak of another bird in a different position. We pass both images through the **diffusion model UNet** and extract their features. We then select the **feature vector** corresponding to the **handle point** and compute the **element-wise similarity** between the target image feature representation and our feature vector. With some **thresholding**, we can see that the beak can be exclusively identified using this approach.*

</div>

## Dragging the Identified part of the image
Now that we have a method to identify the part of the image corresponding to the handle point, our next objective is to move this part so that the handle point aligns with the target point. To achieve this, we incorporate additional guidance terms based on the [Self-Guidance](https://arxiv.org/abs/2306.00986) paper.

Our approach is straightforward: using the heatmap generated during the process, we calculate the average of the spatial locations, weighted by the similarity score. This calculation yields the **centroid** of the part to be moved in a differentiable manner. At the starting point, this centroid is approximately aligned with the handle point.

We then establish a simple objective: to minimize the **L2 distance** between the centroid and the target point. This objective serves as a guiding mechanism for the score predicted by the diffusion model. This process is clearly illustrated below,

Given a spatial heat map `H(x, y)` where each pixel value represents the intensity of heat, we can compute the **centroid** of the heat distribution using a weighted average of the spatial locations. The computation can be expressed as follows:

\begin{equation}
C = \frac{\sum_{x,y} (x, y) \cdot H(x, y)}{\sum_{x,y} H(x, y)}
\end{equation}

Where:
- **C** is the centroid of the heat map.
- **(x, y)** represents the spatial coordinates.
- **H(x, y)** is the heat value at the spatial location `(x, y)`.




