close all;
clear;
clc;

load('digits.mat');
% VaryingArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
% VaryingArray = [100, 200, 400, 784, 1600, 2000, 3000]; % 4000, 5000, 6000, 8000, 10000];
VaryingArray = [1];
VA = [];
AccArray = [];
for scriptI = 1:size(VaryingArray,2)
    % Getting a reduced number of training images to train on
    trainingSize = 5000;
    [TrainingImgs , TrainingArray] = SelectTrainingSamples(trainImages, trainingSize);
    TrainingImgsLabels = (trainLabels (:, TrainingArray));

    TrainingImgHogFeatures = [];
    for i = 1:size(TrainingImgs, 2)
        ImgData = TrainingImgs(:,i);
        ImgPixels = reshape(ImgData, [28,28]);
        ImgHogFeatures = extractHOGFeatures(ImgPixels, 'CellSize', [5 5]);
        TrainingImgHogFeatures = [TrainingImgHogFeatures; ImgHogFeatures];
    end
    % Rows are the images.. columns are the features
    
    SVMTrainingImgs = double(TrainingImgHogFeatures);
%     SVMTrainingImgs = (double(TrainingImgs))';
    SVMTrainingImgLabels = (double(TrainingImgsLabels))'; % Rows are the labels.. only 1 column
    
    SVMParams = templateSVM('KernelFunction', 'Polynomial');
    SVMModel = fitcecoc(SVMTrainingImgs, SVMTrainingImgLabels, 'Learners', SVMParams, 'Coding', 'onevsall');
    
    sizeTest = size(testImages, 4);
    CompleteTestSet = SelectTrainingSamples(testImages, sizeTest);
    HardTestSet = CompleteTestSet(:,1:5000);
    EasyTestSet = CompleteTestSet(:,5001:10000);
    HardTestLabels = testLabels(1, 1:5000);
    EasyTestLabels = testLabels(1, 5001:10000);
    
    TestSet = CompleteTestSet;
    TestLabels = testLabels;
    
    TestImgHogFeatures = [];
    for i = 1:size(TestSet, 2)
        ImgData = TestSet(:,i);
        ImgPixels = reshape(ImgData, [28,28]);
        ImgHogFeatures = extractHOGFeatures(ImgPixels, 'CellSize', [5 5]);
        TestImgHogFeatures = [TestImgHogFeatures; ImgHogFeatures];
    end
    
    SVMTestData = double(TestImgHogFeatures);
%     SVMTestData = (double(TestSet))';
    SVMTestLabels = predict(SVMModel, SVMTestData);
    
    Accuracy = CalculateAccuracy(SVMTestLabels, TestLabels');
    AccArray = [AccArray, Accuracy];
end

% plot(VaryingArray, AccArray)


% I = reshape(EVecReduced(:,1) * 255, [28,28]);
% for j = 2:6
%     I = [I reshape(EVecReduced(:,j) * 255, [28,28])];
% end
% I = mat2gray(I);
% imshow(I);

% I = reshape(ReclaimedTrainImgs(:,1), [28,28]);
% I = mat2gray(I);
% imshow(I);