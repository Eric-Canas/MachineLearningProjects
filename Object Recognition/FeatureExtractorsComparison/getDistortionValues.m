function [distortionValue] = getDistortionValues(distortion)

none_values = ["-"];
scale_values = [0.5, 1.5];
rotation_values = [45, 90, 135];
blur_values = [1, 2, 3];
intensity_values = [0.25, 0.5, 0.75];
contrast_values = [0. 1.; 0. 0.5; 0.5 1.];
projection_values = [10, 20, 30];

switch distortion
    case "None"
        distortionValue = none_values;
    case "Scaling"
        distortionValue = scale_values;
    case "Rotation"
        distortionValue = rotation_values;
    case "Blur"
        distortionValue = blur_values;
    case "Projection"
        distortionValue = projection_values;
    case "Intensity"
        distortionValue = intensity_values;
    case "Contrast"
        distortionValue = contrast_values;                                
end

