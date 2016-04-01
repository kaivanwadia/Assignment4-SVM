function [ Accuracy ] = CalculateAccuracy(SVMLabels, SolutionLabels)
%CALCULATEACCURACY Summary of this function goes here
%   Calculate the accuracy of the projected test images based on euclidean
%   distance using k-nn algorithm

error = 0;

for i = 1:size(SVMLabels, 1)
    svmLabel = SVMLabels(i);
    solLabel = SolutionLabels(i);
    if svmLabel ~= solLabel
        error = error + 1;
    end
end
Accuracy = 1 - error/size(SVMLabels, 1);
end

