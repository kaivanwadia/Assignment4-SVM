function [ Mean, EVecs, EValues ] = hw1FindEigendigits( TrainingMatA )
%FINDEIGENDIGITS Summary of this function goes here
%   trainingSize - Number of training images to use for the calculation of the eigenvectors.
%   k - The number of eigenvectors to return. k < trainingSize. 
    Mean = mean(TrainingMatA,2);
    [NumFreedoms, trainingSize] = size(TrainingMatA);
    NumFreedoms = 65000;
    if trainingSize <= NumFreedoms
        aMat = bsxfun(@minus, double(TrainingMatA), Mean);
        %Calculating the covariance matrix
        CovarianceMat = (aMat' * aMat)/trainingSize;
        % The next few lines calculate the eigen values and vectors and sorts
        % them and then normalizes the vectors
        [EVecs, EValuesMat] = eig(CovarianceMat);
        EValues = diag(EValuesMat);
        [EValues, SortIndex] = sort(EValues, 'descend');
        EVecs = EVecs(:,SortIndex);
        EVecs = double(TrainingMatA) * double(EVecs);
        EVecs = normc(EVecs);
    else
        aMat = bsxfun(@minus, double(TrainingMatA), Mean);
        %Calculating the covariance matrix
        CovarianceMat = (aMat * aMat')/trainingSize;
        % The next few lines calculate the eigen values and vectors and sorts
        % them and then normalizes the vectors
        [EVecs, EValuesMat] = eig(CovarianceMat);
        EValues = diag(EValuesMat);
        [EValues, SortIndex] = sort(EValues, 'descend');
        EVecs = EVecs(:,SortIndex);
        EVecs = normc(EVecs);
    end
end

