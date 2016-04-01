close all;
clear;
clc;

load('digits.mat');
% VaryingArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
% VaryingArray = [100, 200, 400, 784, 1600, 2000, 3000, 4000, 5000, 6000, 8000, 10000];
VaryingArray = [1];
AccArray = [];
for scriptI = 1:size(VaryingArray,2)
    % Getting a reduced number of training images to train on
    trainingSize = 1000;
    [TrainingImgs , TrainingArray] = SelectTrainingSamples(trainImages, trainingSize);
    TrainingImgsLabels = (trainLabels (:, TrainingArray));
    
%     Get the Mean matrix and the Eigen vectors of the reduced training images
%     [Mean, EVectors, EValues] = hw1FindEigendigits(TrainingImgs);
%     currentSum = 0;
%     eigenSum = sum(EValues);
%     numEigen = 0;
%     for i = 1:size(EValues, 1)
%         eigenValue = EValues(i,1);
%         currentSum = currentSum + eigenValue;
%         if (currentSum/eigenSum > 0.99)
%             numEigen = i;
%             break;
%         end
%     end
%     EVecReduced = EVectors(:,1:150);
%     EVecReduced = EVectors(:,1:numEigen);

%     numTrainImgsToClassify = trainingSize;
%     [TrainImgsToClassify, TrainingArrayClassify] = SelectTrainingSamples(trainImages, numTrainImgsToClassify);
%     AMat = bsxfun(@minus, double(TrainImgsToClassify), Mean);
%     ProjectedTrainingImgs = (EVecReduced')*double(AMat);
%     ReclaimedTrainImgs = EVecReduced * ProjectedTrainingImgs;
    
%     ProjectedTestImgs = (EVecReduced') * double(HardTestSet);
%     ProjectedTestLabels = HardTestLabels;
%     ReclaimedTestImgs =  EVecReduced * ProjectedTestImgs;
%     K = 3;
%     [Accuracy , testImgLabels] = CalculateAccuracy(ProjectedTrainingImgs', ProjectedTestImgs', TrainingArrayClassify, trainLabels, ProjectedTestLabels, K);
%     AccArray = [AccArray, Accuracy];

    TrainingImgHogFeatures = [];
    for i = 1:size(TrainingImgs, 2)
        ImgData = TrainingImgs(:,i);
        ImgPixels = reshape(ImgData, [28,28]);
        ImgHogFeatures = extractHOGFeatures(ImgPixels);
        TrainingImgHogFeatures = [TrainingImgHogFeatures; ImgHogFeatures];
    end
    
    % Rows are the images.. columns are the features
    
    % SVMTrainingImgs = (double(TrainingImgs))';
    SVMTrainingImgs = double(TrainingImgHogFeatures);
    SVMTrainingImgLabels = (double(TrainingImgsLabels))'; % Rows are the labels.. only 1 column
    
    SVMParams = templateSVM('KernelFunction', 'linear');
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
        ImgHogFeatures = extractHOGFeatures(ImgPixels);
        TestImgHogFeatures = [TestImgHogFeatures; ImgHogFeatures];
    end
    
    % SVMTestData = (double(TestSet))';
    SVMTestData = double(TestImgHogFeatures);
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