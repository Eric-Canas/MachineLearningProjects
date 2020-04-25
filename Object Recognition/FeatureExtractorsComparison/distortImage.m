function [distortedImg] = distortImage(img, distortion, value)

%Applying
switch distortion
    case "None"
        distortedImg = img;
    case "Scaling"
        distortedImg = imresize(img, value);
    case "Rotation"
        distortedImg = imrotate(img, value);
    case "Blur"
        distortedImg = imgaussfilt(img, value);
    case "Projection"
        defaultProjectionMatrix = projective2d([cosd(value) -sind(value) 0.0001;...
                                        sind(value) cosd(value) 0.001; 0 0 1]);
        distortedImg = imwarp(img, defaultProjectionMatrix);
    case "Intensity"
        distortedImg = imlocalbrighten(img, value);
    case "Contrast"
        distortedImg = imadjust(img, value);                                
end
