function [matchMetric] = matchAndShow(img1, img2, features1, features2,...
                                        validPoints1, validPoints2,... 
                                        title_of_plot, folder, value)                                    
[indexPairs, matchMetric] = matchFeatures(features1,features2);
matchedPoints1 = validPoints1(indexPairs(:,1),:);
matchedPoints2 = validPoints2(indexPairs(:,2),:);

fig = figure(1);
%Show the connections
subplot(2,1,1);
showMatchedFeatures(img1,img2, matchedPoints1, matchedPoints2, 'montage');
title(title_of_plot+" ["+value+"]")

%Shows the overlapped images with connections
subplot(2,2,3);
showMatchedFeatures(img1,img2, matchedPoints1, matchedPoints2);%, 'montage');
save_at = "results/"+folder+"/Matches-"+size(matchMetric,1)+"-"+title_of_plot+".png";
title_of_plot = "Mean SSD/Hamming: "+ mean(matchMetric)...
    +". Matches: "+ size(matchMetric,1);
title(title_of_plot)

%Try to reconstruct the image
subplot(2,2,4);
try
    [tform,~,~] = estimateGeometricTransform(matchedPoints2, matchedPoints1,...
                                             'affine');
    imgReconstructed = imwarp(img2,tform);
catch
    imgReconstructed = zeros(size(img1));
end
imshow(imgReconstructed);
title("Reconstruction")
if ~exist("results/"+folder, 'dir')
       mkdir("results/"+folder)
end
saveas(fig, save_at);
end