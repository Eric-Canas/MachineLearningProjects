function [matchMetric] = matchAndShow(img1, img2, features1, features2,...
                                        validPoints1, validPoints2,... 
                                        title_of_plot)

[indexPairs, matchMetric] = matchFeatures(features1,features2);
matchedPoints1 = validPoints1(indexPairs(:,1),:);
matchedPoints2 = validPoints2(indexPairs(:,2),:);
showMatchedFeatures(img1,img2, matchedPoints1, matchedPoints2, 'montage');
title(title_of_plot)
end

