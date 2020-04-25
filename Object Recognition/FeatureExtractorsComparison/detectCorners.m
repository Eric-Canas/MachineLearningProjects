function [corners, img_tagged] =  detectCorners(img, algorithm)
% Take the corners for an specified algorithm

% Put the image in grayscale if it is not.
if size(img, 3) > 1
    img = rgb2gray(img);
end

% Apply the algorithm
switch algorithm
    case "FAST"
        corners = detectFASTFeatures(img);
    case "MinEigen"
        corners = detectMinEigenFeatures(img);
    case "Harris"
        corners = detectHarrisFeatures(img);
    case "SURF"
        corners = detectSURFFeatures(img);
    case "KAZE"
        corners = detectKAZEFeatures(img);
    case "BRISK"
        corners = detectBRISKFeatures(img);
    case "MSER"
        corners = detectMSERFeatures(img);
    case "ORB"
        corners = detectORBFeatures(img);
    otherwise
        fprintf("Non recognized descriptor")
end
    %corners.selectStrongest(points_to_show)
    img_tagged = insertMarker(img,corners,'circle');
end