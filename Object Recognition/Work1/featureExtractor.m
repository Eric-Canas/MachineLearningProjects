function [features, validPoints] =  featureExtractor(img, corners, method)
% Extract the features of a given corners

% Put the image in grayscale if it is not.
if size(img, 3) > 1
    img = rgb2gray(img);
end

% Apply the extractor
[features, validPoints] = extractFeatures(img, corners, 'Method', method);

end

