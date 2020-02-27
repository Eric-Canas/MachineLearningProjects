%Descriptors to use and compare
descriptors = ["FAST", "MinEigen", "Harris", "SURF", "KAZE",...
                "BRISK", "MSER"];%, "ORB"]; ORB only works with ORB descriptors
%Feature extractors to use and compare
features = ["SURF", "KAZE", "FREAK", "BRISK", "Block"]; %"ORB", ;
%Distorsions to be applied
distortions = ["Rotation", "Scaling", "Projection", "Blurring",...
                "Intensity", "Contrast"];
%Images where try it
files = dir('images/');
files = files(3:size(files)).name;
for image_path = files
    sprintf(image_path)
    img = imread("images/"+image_path);
    for distortion = distortions
        distorted_img = distortImage(img, distortion);
        for descriptor = descriptors
            [corners, img_tagged] = detectCorners(img, descriptor);
            for feature = features
                [feat_img, validPointsImg] = featureExtractor(img, corners, feature);
                [feat_dist, validPointsDist] = featureExtractor(distorted_img, corners, feature);
                title = descriptor+" Descriptors with "+feature+...
                " Features. Distortion "+distortion+". "+... 
                "[Press any key to go next]";
                [matchMetric] = matchAndShow(img, distorted_img,...
                                            feat_img,feat_dist,...
                                            validPointsImg, validPointsDist,...
                                            title);
                
                waitforbuttonpress;
            end
        end
    end
end