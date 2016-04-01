function [ SelectedTrainingImages, TrainingArray ] = SelectTrainingSamples( TrainingDataComplete , NumSamples )
%SELECTTRAININGSAMPLES Summary of this function goes here
% Function to select which training samples to take
% The function initially reshapes the data into the form required i.e.
% numOfPixels * numOfImages. We then select the required number of training
% images and return the data of those images
    ReshapedData = squeeze(TrainingDataComplete);
    [dim1, dim2, totImgs] = size(ReshapedData);
    ReshapedData = reshape(ReshapedData, [dim1*dim2, totImgs]);
    TrainingArray = (1:NumSamples);
    SelectedTrainingImages = ReshapedData(:,TrainingArray);
end

