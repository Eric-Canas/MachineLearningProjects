%% OR Lab1: Landmark detection and descriptors

% NOTE: ORB descriptors only work with ORB feature extractors

% Set the list of descriptors, features and distortions
descriptors = "ORB"; %["FAST", "MinEigen", "Harris", "SURF", "KAZE", "BRISK", "MSER"];% "ORB";
features = "ORB"; %["SURF", "KAZE", "FREAK", "BRISK", "Block"];% "ORB";
distortions = ["None", "Rotation", "Scaling"]; %["None", "Rotation", "Scaling", "Projection", "Blur", "Intensity", "Contrast"];

scale_values = [0.5, 1.5];
rotation_values = [45, 90, 135];
blur_values = [1, 2, 3];
intensity_values = [0.25, 0.5, 0.75];
contrast_values = [0. 1.; 0. 0.5; 0.5 1.];
projection_values = [10, 20, 30];

% Set the dataset directory
files = dir('images/');

% Get all the combinatory results
for n = 3:size(dir('images/')) % Iterate over the dataset
    img = imread("images/" + files(n).name); % Original image
    
    for distortion = distortions % Iterate over all the distortions
        
        for value = getDistortionValues(distortion)% Iterate over all values for this distortion
            distorted_img = distortImage(img, distortion, value); % Distorted image

            for descriptor = descriptors % Iterate over all feature descriptors
                [corners, img_tagged] = detectCorners(img, descriptor);

                for feature = features % Iterate over all feature extractors

                    % Obtain the features of the original and distorted images
                    [feat_img, validPointsImg] = featureExtractor(img, corners, feature);
                    [feat_dist, validPointsDist] = featureExtractor(distorted_img, corners, feature);

                    title = descriptor + " Descriptors with " + feature + ...
                    " Features. Distortion " + distortion + ". ";

                    % RANSAC
                    [matchMetric] = matchAndShow(img, distorted_img, feat_img, feat_dist,...
                                                validPointsImg, validPointsDist,...
                                                title, distortion, value);

                    sprintf(title + " Saved"); % Save the file
                end
            end
        end
    end
end