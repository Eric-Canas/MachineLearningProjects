function [distortedImg] = distortImage(img, distortion)
% Apply a distortion over a given image
% Default parameters
scalingDefaultResize = 1.5;
defaultRotation = 45;
defaultGaussianSigma = 2;
theta = 10;
defaultProjectionMatrix = projective2d([cosd(theta) -sind(theta) 0.0001;...
                                        sind(theta) cosd(theta) 0.001;...
                                        0 0 1]);

%Applying
switch distortion
    case "Scaling"
        distortedImg = imresize(img, scalingDefaultResize);
    case "Rotation"
        distortedImg = imrotate(img, defaultRotation);
    case "Blurring"
        distortedImg = imgaussfilt(img, defaultGaussianSigma);
    case "Projection"
        distortedImg = imwarp(img, defaultProjectionMatrix);
    case "Intensity"
        distortedImg = imlocalbrighten(img);
    case "Contrast"
        distortedImg = imadjust(img,[0., 1.]);
                                
end

