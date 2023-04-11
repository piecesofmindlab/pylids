<p align="center">
  <img src="pylids_logo.png"/>
</p>

# Pylids
A suite of tools for robust and generalizable estimation of eye shape from videos.

With pylids you can use a pretrained DNN model based on [DLC](https://github.com/DeepLabCut/DeepLabCut) to-

* Estimate the pupil outline 
* Estimate the shape of the eylids

pylids also provides users with tools to finetune the default DNN model to ensure generalization on their dataset. Users can - 

* Automatically generate optimally selected domain specific data augmentations to improve pupil and eyelid estimation on new datasets
* Select miniminum frames to relabel from the new dataset to ensure generalization

pylids has been built to be used with the pupil lab gaze estimation pipeline.

![](pylids_readme.gif)

### How to install pylids

Use the shell script pylids_setup.sh
Tested with CUDA version 10.0

### Demos

Check out the notebook in the demo folder
